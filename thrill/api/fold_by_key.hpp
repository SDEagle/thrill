/*******************************************************************************
 * thrill/api/reduce_by_key.hpp
 *
 * DIANode for a reduce operation. Performs the actual reduce operation
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2015 Matthias Stumpp <mstumpp@gmail.com>
 * Copyright (C) 2015 Alexander Noe <aleexnoe@gmail.com>
 * Copyright (C) 2015 Sebastian Lamm <seba.lamm@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_FOLD_BY_KEY_HEADER
#define THRILL_API_FOLD_BY_KEY_HEADER

#include <thrill/api/dia.hpp>
#include <thrill/api/dop_node.hpp>
#include <thrill/common/functional.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/common/meta.hpp>
#include <thrill/common/porting.hpp>
#include <thrill/core/fold_by_hash_post_phase.hpp>
#include <thrill/core/location_detection.hpp>

#include <functional>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace thrill {
namespace api {

class DefaultFoldConfig : public core::DefaultFoldConfig
{ };

/*!
 * A DIANode which performs a Reduce operation. Reduce groups the elements in a
 * DIA by their key and reduces every key bucket to a single element each. The
 * ReduceNode stores the key_extractor and the fold_function UDFs. The
 * chainable LOps ahead of the Reduce operation are stored in the Stack. The
 * ReduceNode has the type ValueType, which is the result type of the
 * fold_function.
 *
 * \tparam ValueType Output type of the Reduce operation
 * \tparam Stack Function stack, which contains the chained lambdas between the
 *  last and this DIANode.
 * \tparam KeyExtractor Type of the key_extractor function.
 * \tparam FoldFunction Type of the fold_function.
 * \tparam VolatileKey Whether to reuse the key once extracted in during pre reduce
 * (false) or let the post reduce extract the key again (true).
 *
 * \ingroup api_layer
 */
template <typename ValueType,
          typename KeyExtractor, typename FoldFunction,
          typename FoldConfig,
          typename KeyHashFunction, typename KeyEqualFunction,
          bool UseLocationDetection>
class FoldNode final : public DOpNode<ValueType>
{
private:
    static constexpr bool debug = false;

    using Super = DOpNode<ValueType>;
    using Super::context_;

    using Key = typename common::FunctionTraits<KeyExtractor>::result_type;
    using ValueIn =
              typename common::FunctionTraits<KeyExtractor>::template arg_plain<0>;

    using TableItem = std::pair<Key, ValueType>;

    using HashIndexFunction = core::ReduceByHash<Key, KeyHashFunction>;

    static constexpr bool use_mix_stream_ = FoldConfig::use_mix_stream_;
    static constexpr bool use_post_thread_ = FoldConfig::use_post_thread_;

    //! Emitter for PostPhase to push elements to next DIA object.
    class Emitter
    {
    public:
        explicit Emitter(FoldNode* node) : node_(node) { }
        void operator () (const ValueType& item) const
        { return node_->PushItem(item); }

    private:
        FoldNode* node_;
    };

    class HashCount
    {
    public:
        using HashType = size_t;
        using CounterType = uint8_t;

        size_t hash;
        CounterType count;

        static constexpr size_t counter_bits_ = 8 * sizeof(CounterType);

        HashCount operator + (const HashCount& b) const {
            assert(hash == b.hash);
            return HashCount { hash, common::AddTruncToType(count, b.count) };
        }

        HashCount& operator += (const HashCount& b) {
            assert(hash == b.hash);
            count = common::AddTruncToType(count, b.count);
            return *this;
        }

        bool operator < (const HashCount& b) const { return hash < b.hash; }

        //! method to check if this hash count should be broadcasted to all
        //! workers interested -- for GroupByKey -> always.
        bool NeedBroadcast() const {
            return true;
        }

        //! Read count from BitReader
        template <typename BitReader>
        void ReadBits(BitReader& reader) {
            count = reader.GetBits(counter_bits_);
        }

        //! Write count and dia_mask to BitWriter
        template <typename BitWriter>
        void WriteBits(BitWriter& writer) const {
            writer.PutBits(count, counter_bits_);
        }
    };

public:
    /*!
     * Constructor for a FoldNode. Sets the parent, stack, key_extractor and
     * fold_function.
     */
    template <typename ParentDIA>
    FoldNode(const ParentDIA& parent,
               const char* label,
               const KeyExtractor& key_extractor,
               const FoldFunction& fold_function,
               const FoldConfig& config,
               const KeyHashFunction& key_hash_function,
               const KeyEqualFunction& key_equal_function)
        : Super(parent.ctx(), label, { parent.id() }, { parent.node() }),
          key_extractor_(key_extractor),
          fold_function_(fold_function),
          hash_function_(key_hash_function),
          location_detection_(parent.ctx(), Super::id()),
          pre_file_(context_.GetFile(this)),
          mix_stream_(use_mix_stream_ ?
                      parent.ctx().GetNewMixStream(this) : nullptr),
          cat_stream_(use_mix_stream_ ?
                      nullptr : parent.ctx().GetNewCatStream(this)),
          emitters_(use_mix_stream_ ?
                    mix_stream_->GetWriters() : cat_stream_->GetWriters()),
          post_phase_(
              context_, Super::id(), key_extractor, fold_function,
              Emitter(this), config,
              HashIndexFunction(key_hash_function), key_equal_function)
    {
        // Hook PreOp: Locally hash elements of the current DIA onto buckets and
        // reduce each bucket to a single value, afterwards send data to another
        // worker given by the shuffle algorithm.
        auto pre_op_fn = [this](const ValueIn& input) {
                             PreOp(input);
                         };
        // close the function stack with our pre op and register it at
        // parent node for output
        auto lop_chain = parent.stack().push(pre_op_fn).fold();
        parent.node()->AddChild(this, lop_chain);
    }

    DIAMemUse PreOpMemUse() final {
        // request maximum RAM limit, the value is calculated by StageBuilder,
        // and set as DIABase::mem_limit_.
        return DIAMemUse::Max();
    }

    void StartPreOp(size_t /* id */) final {
        LOG << *this << " running StartPreOp";
        pre_writer_ = pre_file_.GetWriter();
        if (!use_post_thread_) {
            // use pre_phase without extra thread
            // pre_phase_.Initialize(DIABase::mem_limit_);
            if (UseLocationDetection)
                location_detection_.Initialize(DIABase::mem_limit_);
        }
        else {
            if (UseLocationDetection)
                location_detection_.Initialize(DIABase::mem_limit_ / 2);
            post_phase_.Initialize(DIABase::mem_limit_ / 2);

            // start additional thread to receive from the channel
            thread_ = common::CreateThread([this] { ProcessChannel(); });
        }
    }

    //! Send all elements to their designated PEs
    void PreOp(const ValueIn& v) {
        size_t hash = hash_function_(key_extractor_(v));
        if (UseLocationDetection) {
            pre_writer_.Put(v);
            location_detection_.Insert(HashCount { hash, 1 });
        }
        else {
            const size_t recipient = hash % emitters_.size();
            emitters_[recipient].Put(v);
        }
    }

    void StopPreOp(size_t /* id */) final {
        LOG << *this << " running StopPreOp";
        pre_writer_.Close();
        // Flush hash table before the postOp
        if (UseLocationDetection) {
            std::unordered_map<size_t, size_t> target_processors;
            size_t max_hash = location_detection_.Flush(target_processors);
            auto file_reader = pre_file_.GetConsumeReader();
            while (file_reader.HasNext()) {
                ValueIn in = file_reader.template Next<ValueIn>();
                Key key = key_extractor_(in);

                size_t hr = hash_function_(key) % max_hash;
                auto target_processor = target_processors.find(hr);
                emitters_[target_processor->second].Put(in);
            }
        }
        // data has been pushed during pre-op -> close emitters
        for (size_t i = 0; i < emitters_.size(); i++)
            emitters_[i].Close();
        // pre_phase_.FlushAll();
        // pre_phase_.CloseAll();
        // waiting for the additional thread to finish the reduce
        location_detection_.Dispose();
        if (use_post_thread_) thread_.join();
        use_mix_stream_ ? mix_stream_->Close() : cat_stream_->Close();
    }

    void Execute() final { }

    DIAMemUse PushDataMemUse() final {
        return DIAMemUse::Max();
    }

    void PushData(bool consume) final {
        if (!use_post_thread_ && !folded_) {
            // not final reduced, and no additional thread, perform post reduce
            post_phase_.Initialize(DIABase::mem_limit_);
            ProcessChannel();

            folded_ = true;
        }
        post_phase_.PushData(consume);
    }

    //! process the inbound data in the post reduce phase
    void ProcessChannel() {
        if (use_mix_stream_)
        {
            auto reader = mix_stream_->GetMixReader(/* consume */ true);
            sLOG << "reading data from" << mix_stream_->id()
                 << "to push into post phase which flushes to" << this->id();
            while (reader.HasNext()) {
                post_phase_.Insert(reader.template Next<ValueIn>());
            }
        }
        else
        {
            auto reader = cat_stream_->GetCatReader(/* consume */ true);
            sLOG << "reading data from" << cat_stream_->id()
                 << "to push into post phase which flushes to" << this->id();
            while (reader.HasNext()) {
                post_phase_.Insert(reader.template Next<ValueIn>());
            }
        }
    }

    void Dispose() final {
        post_phase_.Dispose();
    }

private:
    KeyExtractor key_extractor_;
    FoldFunction fold_function_;
    KeyHashFunction hash_function_;

    core::LocationDetection<HashCount> location_detection_;

    //! location detection and associated files
    data::File pre_file_;
    data::File::Writer pre_writer_;

    // pointers for both Mix and CatStream. only one is used, the other costs
    // only a null pointer.
    data::MixStreamPtr mix_stream_;
    data::CatStreamPtr cat_stream_;

    std::vector<data::Stream::Writer> emitters_;
    //! handle to additional thread for post phase
    std::thread thread_;

    core::FoldByHashPostPhase<
        ValueType, ValueIn, Key,
        KeyExtractor, FoldFunction, Emitter,
        FoldConfig, HashIndexFunction, KeyEqualFunction> post_phase_;

    bool folded_ = false;
};

// template <typename ValueType, typename Stack>
// template <typename ValueOut,
//           typename KeyExtractor, typename FoldFunction,
//           typename FoldConfig>
// auto DIA<ValueType, Stack>::FoldByKey(
//     const KeyExtractor &key_extractor,
//     const FoldFunction &fold_function,
//     const FoldConfig& fold_config) const {
//     // forward to main function
//     using Key = typename common::FunctionTraits<KeyExtractor>::result_type;
//     return FoldByKey(
//         NoDuplicateDetectionTag,
//         key_extractor, fold_function, fold_config,
//         std::hash<Key>(), std::equal_to<Key>());
// }

// template <typename ValueType, typename Stack>
// template <typename ValueOut,
//           typename KeyExtractor, typename FoldFunction,
//           typename FoldConfig, typename KeyHashFunction>
// auto DIA<ValueType, Stack>::FoldByKey(
//     const KeyExtractor &key_extractor,
//     const FoldFunction &fold_function,
//     const FoldConfig &fold_config,
//     const KeyHashFunction &key_hash_function) const {
//     // forward to main function
//     using Key = typename common::FunctionTraits<KeyExtractor>::result_type;
//     return FoldByKey(
//         NoDuplicateDetectionTag,
//         key_extractor, fold_function, fold_config,
//         key_hash_function, std::equal_to<Key>());
// }

// template <typename ValueType, typename Stack>
// template <typename ValueOut,
//           typename KeyExtractor, typename FoldFunction, typename FoldConfig,
//           typename KeyHashFunction, typename KeyEqualFunction>
// auto DIA<ValueType, Stack>::FoldByKey(
//     const KeyExtractor &key_extractor,
//     const FoldFunction &fold_function,
//     const FoldConfig &fold_config,
//     const KeyHashFunction &key_hash_function,
//     const KeyEqualFunction &key_equal_funtion) const {
//     // forward to main function
//     return FoldByKey(
//         NoDuplicateDetectionTag,
//         key_extractor, fold_function, fold_config,
//         key_hash_function, key_equal_funtion);
// }

template <typename ValueType, typename Stack>
template <typename ValueOut,
          bool DuplicateDetectionValue,
          typename KeyExtractor, typename FoldFunction, typename FoldConfig,
          typename KeyHashFunction, typename KeyEqualFunction>
auto DIA<ValueType, Stack>::FoldByKey(
    const DuplicateDetectionFlag<DuplicateDetectionValue>&,
    const KeyExtractor &key_extractor,
    const FoldFunction &fold_function,
    const FoldConfig &fold_config,
    const KeyHashFunction &key_hash_function,
    const KeyEqualFunction &key_equal_funtion) const {
    assert(IsValid());

    // static_assert(
    //     std::is_convertible<
    //         ValueOut,
    //         typename common::FunctionTraits<FoldFunction>::template arg<0>
    //         >::value,
    //     "FoldFunction has the wrong input type for the accumulator arg");

    static_assert(
        std::is_convertible<
            ValueType,
            typename common::FunctionTraits<FoldFunction>::template arg<1>
            >::value,
        "FoldFunction has the wrong input type");

    // static_assert(
    //     std::is_same<
    //         ValueOut,
    //         typename common::FunctionTraits<FoldFunction>::result_type>::value,
    //     "FoldFunction has the wrong output type");

    static_assert(
        std::is_same<
            typename std::decay<typename common::FunctionTraits<KeyExtractor>::
                                template arg<0> >::type,
            ValueType>::value,
        "KeyExtractor has the wrong input type");

    using Key = typename common::FunctionTraits<KeyExtractor>::result_type;

    using FoldNode = api::FoldNode<
              std::pair<Key, ValueOut>, KeyExtractor, FoldFunction, FoldConfig,
              KeyHashFunction, KeyEqualFunction,
              DuplicateDetectionValue>;

    auto node = common::MakeCounting<FoldNode>(
        *this, "FoldByKey",
        key_extractor, fold_function, fold_config,
        key_hash_function, key_equal_funtion);

    return DIA<std::pair<Key, ValueOut>>(node);
}

} // namespace api
} // namespace thrill

#endif // !THRILL_API_REDUCE_BY_KEY_HEADER

/******************************************************************************/
