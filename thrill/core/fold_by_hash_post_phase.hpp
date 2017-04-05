/*******************************************************************************
 * thrill/core/reduce_by_hash_post_phase.hpp
 *
 * Hash table with support for reduce.
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2015 Matthias Stumpp <mstumpp@gmail.com>
 * Copyright (C) 2015 Alexander Noe <aleexnoe@gmail.com>
 * Copyright (C) 2015-2016 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_CORE_FOLD_BY_HASH_POST_PHASE_HEADER
#define THRILL_CORE_FOLD_BY_HASH_POST_PHASE_HEADER

#include <thrill/api/context.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/core/fold_functional.hpp>
#include <thrill/core/fold_probing_hash_table.hpp>
#include <thrill/data/file.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace thrill {
namespace core {

template <typename ValueOut, typename ValueIn, typename Key,
          typename KeyExtractor, typename FoldFunction, typename Emitter,
          typename FoldConfig_ = DefaultFoldConfig,
          typename IndexFunction = ReduceByHash<Key>,
          typename KeyEqualFunction = std::equal_to<Key> >
class FoldByHashPostPhase
{
    static constexpr bool debug = false;

public:
    using FoldConfig = FoldConfig_;
    using PhaseEmitter = FoldPostPhaseEmitter<
              ValueOut, ValueIn, Emitter>;

    using Table = typename FoldTableSelect<
              FoldConfig::table_impl_,
              ValueOut, Key, ValueIn,
              KeyExtractor, FoldFunction, PhaseEmitter,
              FoldConfig, IndexFunction, KeyEqualFunction>::type;

    /*!
     * A data structure which takes an arbitrary value and extracts a key using
     * a key extractor function from that value. Afterwards, the value is hashed
     * based on the key into some slot.
     */
    FoldByHashPostPhase(
        Context& ctx, size_t dia_id,
        const KeyExtractor& key_extractor,
        const FoldFunction& fold_function,
        const Emitter& emit,
        const FoldConfig& config = FoldConfig(),
        const IndexFunction& index_function = IndexFunction(),
        const KeyEqualFunction& key_equal_function = KeyEqualFunction())
        : config_(config),
          emitter_(emit),
          table_(ctx, dia_id,
                 key_extractor, fold_function, emitter_,
                 /* num_partitions */ 32, /* TODO(tb): parameterize */
                 config, /* immediate_flush */ false,
                 index_function, key_equal_function),
          spilling_file_(ctx.GetFile(dia_id)) { }

    //! non-copyable: delete copy-constructor
    FoldByHashPostPhase(const FoldByHashPostPhase&) = delete;
    //! non-copyable: delete assignment operator
    FoldByHashPostPhase& operator = (const FoldByHashPostPhase&) = delete;

    void Initialize(size_t limit_memory_bytes) {
        table_.Initialize(limit_memory_bytes);
        spilling_writer_ = spilling_file_.GetWriter();
    }

    void Insert(const ValueIn& value) {
        if (!table_.Insert(value)) {
            spilling_writer_.Put(value);
        }
    }

    //! Flushes all items in the whole table.
    template <bool DoCache>
    void Flush(bool consume, data::File::Writer* writer = nullptr) {
        LOG << "Flushing items";

        // read primary hash table, since ReduceByHash delivers items in any
        // order, we can just emit items from fully reduced partitions.

        table_.FlushAll(consume, /* grow */ false,
            [this, writer](const size_t& partition_id, const ValueOut& p) {
                if (DoCache) writer->Put(p);
                emitter_.Emit(partition_id, p);
            });

        if (spilling_file_.num_items() == 0) {
            LOG << "Flushed items directly.";
            return;
        }

        table_.Dispose();

        assert(consume && "Items were spilled hence Flushing must consume");

        // if partially reduce files remain, create new hash tables to process
        // them iteratively.

        data::File remaining_items = std::move(spilling_file_);

        size_t iteration = 0;

        while (remaining_items.num_items() > 0)
        {
            data::File next_spill = table_.ctx().GetFile(table_.dia_id());
            data::File::Writer next_writer = next_spill.GetWriter();

            Table subtable(
                table_.ctx(), table_.dia_id(),
                table_.key_extractor(), table_.fold_function(), emitter_,
                /* num_partitions */ 32, config_, /* immediate_flush */ false,
                IndexFunction(iteration, table_.index_function()),
                table_.key_equal_function());

            subtable.Initialize(table_.limit_memory_bytes());

            auto reader = remaining_items.GetConsumeReader();

            while (reader.HasNext()) {
                const ValueIn& value = reader.template Next<ValueIn>();
                if (!subtable.Insert(value)) {
                    next_writer.Put(value);
                }
            }

            subtable.FlushAll(consume, /* grow */ false,
                [this, writer](const size_t& partition_id, const ValueOut& p) {
                    if (DoCache) writer->Put(p);
                    emitter_.Emit(partition_id, p);
                });

            subtable.Dispose();

            next_writer.Close();
            remaining_items = std::move(next_spill);
            ++iteration;
        }

        LOG << "Flushed items";
    }

    //! Push data into emitter
    void PushData(bool consume = false) {
        spilling_writer_.Close();

        if (!cache_)
        {
            if (spilling_file_.num_items() == 0) {
                // all items did fit into the table
                Flush</* DoCache */ false>(consume);
            }
            else {
                // we have items left over
                cache_ = table_.ctx().GetFilePtr(table_.dia_id());
                data::File::Writer writer = cache_->GetWriter();
                Flush</* DoCache */ true>(true, &writer);
            }
        }
        else
        {
            // previous PushData() has stored data in cache_
            data::File::Reader reader = cache_->GetReader(consume);
            while (reader.HasNext())
                emitter_.Emit(reader.Next<ValueOut>());
        }
    }

    void Dispose() {
        table_.Dispose();
        if (cache_) cache_.reset();
    }

    //! \name Accessors
    //! \{

    //! Returns mutable reference to first table_
    Table& table() { return table_; }

    //! Returns the total num of items in the table.
    size_t num_items() const { return table_.num_items(); }

    //! \}

private:
    //! Stored reduce config to initialize the subtable.
    FoldConfig config_;

    //! Emitters used to parameterize hash table for output to next DIA node.
    PhaseEmitter emitter_;

    //! the first-level hash table implementation
    Table table_;

    //! File for storing data in-case we need multiple re-reduce levels.
    data::FilePtr cache_;

    data::File spilling_file_;
    data::File::Writer spilling_writer_;
};

} // namespace core
} // namespace thrill

#endif // !THRILL_CORE_REDUCE_BY_HASH_POST_PHASE_HEADER

/******************************************************************************/
