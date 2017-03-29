/*******************************************************************************
 * thrill/core/reduce_probing_hash_table.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2015 Matthias Stumpp <mstumpp@gmail.com>
 * Copyright (C) 2016 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_CORE_FOLD_PROBING_HASH_TABLE_HEADER
#define THRILL_CORE_FOLD_PROBING_HASH_TABLE_HEADER

#include <thrill/core/fold_functional.hpp>
#include <thrill/core/fold_table.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace thrill {
namespace core {

/*!
 * A data structure which takes an arbitrary value and extracts a key using a
 * key extractor function from that value. A key may also be provided initially
 * as part of a key/value pair, not requiring to extract a key.
 *
 * Afterwards, the key is hashed and the hash is used to assign that key/value
 * pair to some slot.
 *
 * In case a slot already has a key/value pair and the key of that value and the
 * key of the value to be inserted are them same, the values are reduced
 * according to some reduce function. No key/value is added to the data
 * structure.
 *
 * If the keys are different, the next slot (moving to the right) is considered.
 * If the slot is occupied, the same procedure happens again (know as linear
 * probing.)
 *
 * Finally, the key/value pair to be inserted may either:
 *
 * 1.) Be reduced with some other key/value pair, sharing the same key.
 * 2.) Inserted at a free slot.
 * 3.) Trigger a resize of the data structure in case there are no more free
 *     slots in the data structure.
 *
 * The following illustrations shows the general structure of the data
 * structure.  The set of slots is divided into 1..n partitions. Each key is
 * hashed into exactly one partition.
 *
 *
 *     Partition 0 Partition 1 Partition 2 Partition 3 Partition 4
 *     P00 P01 P02 P10 P11 P12 P20 P21 P22 P30 P31 P32 P40 P41 P42
 *    +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *    ||  |   |   ||  |   |   ||  |   |   ||  |   |   ||  |   |  ||
 *    +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *                <-   LI  ->
 *                     LI..Local Index
 *    <-        GI         ->
 *              GI..Global Index
 *         PI 0        PI 1        PI 2        PI 3        PI 4
 *         PI..Partition ID
 *
 */
template <typename TableItem, typename Key, typename ValueIn,
          typename KeyExtractor, typename FoldFunction, typename Emitter,
          typename FoldConfig_,
          typename IndexFunction,
          typename KeyEqualFunction = std::equal_to<Key>>
class FoldProbingHashTable
    : public FoldTable<TableItem, Key, ValueIn,
                         KeyExtractor, FoldFunction, Emitter,
                         FoldConfig_,
                         IndexFunction, KeyEqualFunction>
{
    using Super = FoldTable<TableItem, Key, ValueIn,
                              KeyExtractor, FoldFunction, Emitter,
                              FoldConfig_, IndexFunction,
                              KeyEqualFunction>;
    using Super::debug;
    static constexpr bool debug_items = false;

public:
    using FoldConfig = FoldConfig_;

    FoldProbingHashTable(
        Context& ctx, size_t dia_id,
        const KeyExtractor& key_extractor,
        const FoldFunction& fold_function,
        Emitter& emitter,
        size_t num_partitions,
        const FoldConfig& config = FoldConfig(),
        bool immediate_flush = false,
        const IndexFunction& index_function = IndexFunction(),
        const KeyEqualFunction& key_equal_function = KeyEqualFunction())
        : Super(ctx, dia_id,
                key_extractor, fold_function, emitter,
                num_partitions, config, immediate_flush,
                index_function, key_equal_function)
    { assert(num_partitions > 0); }

    //! Construct the hash table itself. fill it with sentinels. have one extra
    //! cell beyond the end for reducing the sentinel itself.
    void Initialize(size_t limit_memory_bytes) {
        assert(!items_);

        limit_memory_bytes_ = limit_memory_bytes;

        // calculate num_buckets_per_partition_ from the memory limit and the
        // number of partitions required, initialize partition_size_ array.

        assert(limit_memory_bytes_ >= 0 &&
               "limit_memory_bytes must be greater than or equal to 0. "
               "A byte size of zero results in exactly one item per partition");

        num_buckets_per_partition_ = std::max<size_t>(
            1,
            (size_t)(static_cast<double>(limit_memory_bytes_)
                     / static_cast<double>(sizeof(TableItem))
                     / static_cast<double>(num_partitions_)));

        num_buckets_ = num_buckets_per_partition_ * num_partitions_;

        assert(num_buckets_per_partition_ > 0);
        assert(num_buckets_ > 0);

        partition_size_.resize(
            num_partitions_,
            std::min(size_t(config_.initial_items_per_partition_),
                     num_buckets_per_partition_));

        // calculate limit on the number of items in a partition before these
        // are spilled to disk or flushed to network.

        double limit_fill_rate = config_.limit_partition_fill_rate();

        assert(limit_fill_rate >= 0.0 && limit_fill_rate <= 1.0
               && "limit_partition_fill_rate must be between 0.0 and 1.0. "
               "with a fill rate of 0.0, items are immediately flushed.");

        limit_items_per_partition_.resize(
            num_partitions_,
            static_cast<size_t>(
                static_cast<double>(partition_size_[0]) * limit_fill_rate));

        assert(limit_items_per_partition_[0] >= 0);

        // actually allocate the table and initialize the valid ranges, the + 1
        // is for the sentinel's slot.

        items_ = static_cast<TableItem*>(
            operator new ((num_buckets_ + 1) * sizeof(TableItem)));

        for (size_t id = 0; id < num_partitions_; ++id) {
            TableItem* iter = items_ + id * num_buckets_per_partition_;
            TableItem* pend = iter + partition_size_[id];

            for ( ; iter != pend; ++iter)
                new (iter)TableItem();
        }
    }

    ~FoldProbingHashTable() {
        if (items_) Dispose();
    }

    /*!
     * Inserts a value into the table, potentially reducing it in case both the
     * key of the value already in the table and the key of the value to be
     * inserted are the same.
     *
     * An insert may trigger a partial flush of the partition with the most
     * items if the maximal number of items in the table (max_num_items_table)
     * is reached.
     *
     * Alternatively, it may trigger a resize of the table in case the maximal
     * fill ratio per partition is reached.
     *
     * \param value Value to be inserted into the table.
     *
     * \return true if a new key was inserted to the table
     */
    bool Insert(const ValueIn& value) {

        Key item_key = key(value);

        typename IndexFunction::Result h = index_function_(
            item_key, num_partitions_,
            num_buckets_per_partition_, num_buckets_);

        assert(h.partition_id < num_partitions_);


        if (THRILL_UNLIKELY(key_equal_function_(item_key, Key()))) {
            // handle pairs with sentinel key specially by reducing into last
            // element of items.
            TableItem& sentinel = items_[num_buckets_];
            if (sentinel_partition_ == invalid_partition_) {
                // first occurrence of sentinel key
                sentinel = make(item_key, value);
                sentinel_partition_ = h.partition_id;
            }
            else {
                sentinel = fold(std::move(sentinel), value);
                return true;
            }
            ++items_per_partition_[h.partition_id];
            ++num_items_;

            while (THRILL_UNLIKELY(
                       items_per_partition_[h.partition_id] >
                       limit_items_per_partition_[h.partition_id])) {
                GrowAndRehash(h.partition_id);
            }

            return true;
        }

        // calculate local index depending on the current subtable's size
        size_t local_index = h.local_index(partition_size_[h.partition_id]);

        TableItem* pbegin = items_ + h.partition_id * num_buckets_per_partition_;
        TableItem* pend = pbegin + partition_size_[h.partition_id];

        TableItem* begin_iter = pbegin + local_index;
        TableItem* iter = begin_iter;

        while (!key_equal_function_(key(*iter), Key()))
        {
            if (key_equal_function_(key(*iter), item_key))
            {
                *iter = fold(std::move(*iter), value);
                return true;
            }

            ++iter;

            // wrap around if beyond the current partition
            if (THRILL_UNLIKELY(iter == pend))
                iter = pbegin;

            // flush partition and retry, if all slots are reserved
            if (THRILL_UNLIKELY(iter == begin_iter)) {
                return false;
            }
        }

        if (THRILL_UNLIKELY(
                   items_per_partition_[h.partition_id] >=
                   limit_items_per_partition_[h.partition_id])) {
            LOG << "Grow due to "
                << items_per_partition_[h.partition_id] << " >= "
                << limit_items_per_partition_[h.partition_id]
                << " among " << partition_size_[h.partition_id];
            if (!GrowAndRehash(h.partition_id)) {
                return false;
            }
        }

        // insert new pair
        *iter = make(item_key, value);

        // increase counter for partition
        ++items_per_partition_[h.partition_id];
        ++num_items_;

        return true;
    }

    void Relocate(TableItem&& kv) {

        Key item_key = key(kv);

        typename IndexFunction::Result h = index_function_(
            item_key, num_partitions_,
            num_buckets_per_partition_, num_buckets_);

        assert(h.partition_id < num_partitions_);


        if (THRILL_UNLIKELY(key_equal_function_(item_key, Key()))) {
            // handle pairs with sentinel key specially by reducing into last
            // element of items.
            TableItem& sentinel = items_[num_buckets_];
            if (sentinel_partition_ == invalid_partition_) {
                // first occurrence of sentinel key
                sentinel = std::move(kv);
                sentinel_partition_ = h.partition_id;
            }
            else {
                assert(false);
                // sentinel = fold(std::move(sentinel), value);
                return;
            }
            ++items_per_partition_[h.partition_id];
            ++num_items_;

            while (THRILL_UNLIKELY(
                       items_per_partition_[h.partition_id] >
                       limit_items_per_partition_[h.partition_id])) {
                GrowAndRehash(h.partition_id);
            }

            return;
        }

        // calculate local index depending on the current subtable's size
        size_t local_index = h.local_index(partition_size_[h.partition_id]);

        TableItem* pbegin = items_ + h.partition_id * num_buckets_per_partition_;
        TableItem* pend = pbegin + partition_size_[h.partition_id];

        TableItem* begin_iter = pbegin + local_index;
        TableItem* iter = begin_iter;

        while (!key_equal_function_(key(*iter), Key()))
        {
            if (key_equal_function_(key(*iter), item_key))
            {
                assert(false);
                // *iter = fold(std::move(*iter), value);
                return;
            }

            ++iter;

            // wrap around if beyond the current partition
            if (THRILL_UNLIKELY(iter == pend))
                iter = pbegin;

            // flush partition and retry, if all slots are reserved
            if (THRILL_UNLIKELY(iter == begin_iter)) {
                assert(false);
                return;
            }
        }

        if (THRILL_UNLIKELY(
                   items_per_partition_[h.partition_id] >=
                   limit_items_per_partition_[h.partition_id])) {
            LOG << "Grow due to "
                << items_per_partition_[h.partition_id] << " >= "
                << limit_items_per_partition_[h.partition_id]
                << " among " << partition_size_[h.partition_id];
            if (!GrowAndRehash(h.partition_id)) {
                return;
            }
        }

        // insert new pair
        *iter = std::move(kv);

        // increase counter for partition
        ++items_per_partition_[h.partition_id];
        ++num_items_;

        return;
    }

    //! Deallocate items and memory
    void Dispose() {
        if (!items_) return;

        // dispose the items by destructor

        for (size_t id = 0; id < num_partitions_; ++id) {
            TableItem* iter = items_ + id * num_buckets_per_partition_;
            TableItem* pend = iter + partition_size_[id];

            for ( ; iter != pend; ++iter)
                iter->~TableItem();
        }

        if (sentinel_partition_ != invalid_partition_)
            items_[num_buckets_].~TableItem();

        operator delete (items_);
        items_ = nullptr;

        Super::Dispose();
    }

    bool GrowAndRehash(size_t partition_id) {

        size_t old_size = partition_size_[partition_id];
        GrowPartition(partition_id);
        if (partition_size_[partition_id] == old_size) {
            return false;
        }

        // initialize pointers to old range - the second half is still empty
        TableItem* pbegin =
            items_ + partition_id * num_buckets_per_partition_;
        TableItem* iter = pbegin;
        TableItem* pend = pbegin + old_size;

        // reinsert all elements which go into the second partition
        for ( ; iter != pend; ++iter) {
            if (!key_equal_function_(key(*iter), Key())) {
                typename IndexFunction::Result h = index_function_(
                    key(*iter), num_partitions_,
                    num_buckets_per_partition_, num_buckets_);
                if (h.local_index(partition_size_[partition_id]) >= old_size) {
                    --items_per_partition_[partition_id];
                    --num_items_;
                    TableItem item = std::move(*iter);
                    new (iter)TableItem();
                    Relocate(std::move(item));
                }
            }
        }

        iter = pbegin;
        for ( ; iter != pend; ++iter) {
            if (!key_equal_function_(key(*iter), Key())) {
                --items_per_partition_[partition_id];
                --num_items_;
                TableItem item = std::move(*iter);
                new (iter)TableItem();
                Relocate(std::move(item));
            }
        }


        return true;
    }

    //! Grow a partition after a spill or flush (if possible)
    void GrowPartition(size_t partition_id) {

        size_t new_size = (2 * partition_size_[partition_id]);

        if (new_size >= num_buckets_per_partition_ || mem::memory_exceeded)
            return;

        sLOG << "Growing partition" << partition_id
             << "from" << partition_size_[partition_id] << "to" << new_size
             << "limit_items" << new_size * config_.limit_partition_fill_rate();

        // initialize new items

        TableItem* pbegin =
            items_ + partition_id * num_buckets_per_partition_;
        TableItem* iter = pbegin + partition_size_[partition_id];
        TableItem* pend = pbegin + new_size;

        for ( ; iter != pend; ++iter)
            new (iter)TableItem();

        partition_size_[partition_id] = new_size;
        limit_items_per_partition_[partition_id]
            = new_size * config_.limit_partition_fill_rate();
    }

    //! \name Flushing Mechanisms to Next Stage or Phase
    //! \{

    template <typename Emit>
    void FlushPartitionEmit(
        size_t partition_id, bool consume, bool grow, Emit emit) {

        LOG << "Flushing " << items_per_partition_[partition_id]
            << " items of partition: " << partition_id;

        if (sentinel_partition_ == partition_id) {
            emit(partition_id, items_[num_buckets_]);
            if (consume) {
                items_[num_buckets_].~TableItem();
                sentinel_partition_ = invalid_partition_;
            }
        }

        TableItem* iter = items_ + partition_id * num_buckets_per_partition_;
        TableItem* pend = iter + partition_size_[partition_id];

        for ( ; iter != pend; ++iter)
        {
            if (!key_equal_function_(key(*iter), Key())) {
                emit(partition_id, *iter);

                if (consume)
                    *iter = TableItem();
            }
        }

        if (consume) {
            // reset partition specific counter
            num_items_ -= items_per_partition_[partition_id];
            items_per_partition_[partition_id] = 0;
            assert(num_items_ == this->num_items_calc());
        }

        LOG << "Done flushed items of partition: " << partition_id;

        if (grow)
            GrowPartition(partition_id);
    }

    void FlushPartition(size_t partition_id, bool consume, bool grow) {
        FlushPartitionEmit(
            partition_id, consume, grow,
            [this](const size_t& partition_id, const TableItem& p) {
                this->emitter_.Emit(partition_id, p);
            });
    }

    void FlushAll() {
        for (size_t i = 0; i < num_partitions_; ++i) {
            FlushPartition(i, /* consume */ true, /* grow */ false);
        }
    }

    template <typename Emit>
    void FlushAll(bool consume, bool grow, Emit emit) {
        for (size_t i = 0; i < num_partitions_; ++i) {
            FlushPartitionEmit(i, consume, grow, emit);
        }
    }

    //! \}

private:
    using Super::config_;
    using Super::immediate_flush_;
    using Super::index_function_;
    using Super::items_per_partition_;
    using Super::key;
    using Super::key_equal_function_;
    using Super::limit_memory_bytes_;
    using Super::num_buckets_;
    using Super::num_buckets_per_partition_;
    using Super::num_items_;
    using Super::num_partitions_;
    using Super::partition_files_;
    using Super::fold;
    using Super::make;

    //! Storing the actual hash table.
    TableItem* items_ = nullptr;

    //! Current sizes of the partitions because the valid allocated areas grow
    std::vector<size_t> partition_size_;

    //! Current limits on the number of items in a partitions, different for
    //! different partitions, because the valid allocated areas grow.
    std::vector<size_t> limit_items_per_partition_;

    //! sentinel for invalid partition or no sentinel.
    static constexpr size_t invalid_partition_ = size_t(-1);

    //! store the partition id of the sentinel key. implicitly this also stored
    //! whether the sentinel key was found and reduced into
    //! items_[num_buckets_].
    size_t sentinel_partition_ = invalid_partition_;
};

template <typename TableItem, typename Key, typename ValueIn,
          typename KeyExtractor, typename FoldFunction,
          typename Emitter,
          typename FoldConfig, typename IndexFunction,
          typename KeyEqualFunction>
class FoldTableSelect<
        FoldTableImpl::PROBING,
        TableItem, Key, ValueIn, KeyExtractor, FoldFunction,
        Emitter, FoldConfig, IndexFunction, KeyEqualFunction>
{
public:
    using type = FoldProbingHashTable<
              TableItem, Key, ValueIn, KeyExtractor, FoldFunction,
              Emitter, FoldConfig,
              IndexFunction, KeyEqualFunction>;
};

} // namespace core
} // namespace thrill

#endif // !THRILL_CORE_REDUCE_PROBING_HASH_TABLE_HEADER

/******************************************************************************/
