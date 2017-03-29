/*******************************************************************************
 * thrill/core/reduce_functional.hpp
 *
 * Hash table with support for reduce.
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2015 Matthias Stumpp <mstumpp@gmail.com>
 * Copyright (C) 2015 Alexander Noe <aleexnoe@gmail.com>
 * Copyright (C) 2015 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_CORE_FOLD_FUNCTIONAL_HEADER
#define THRILL_CORE_FOLD_FUNCTIONAL_HEADER

#include <thrill/common/defines.hpp>
#include <thrill/common/hash.hpp>
#include <thrill/common/math.hpp>

namespace thrill {
namespace core {

/******************************************************************************/

//! template specialization switch class to convert a Value either to Value
//! (identity) or to a std::pair<Key, Value> with Key generated from Value using
//! a key extractor for VolatileKey implementations.
template <typename Key, typename Input, typename Accumulator>
class FoldMakeTableItem
{
    using TableItem = std::pair<Key, Accumulator>;
public:
    template <typename AccInitializer, typename FoldFunction>
    static TableItem Make(const Key& key, const Input& v,
                            const AccInitializer& init, const FoldFunction& fold) {
        return TableItem(key, fold(init(key), v));
    }

    template <typename KeyExtractor>
    static auto GetKey(const TableItem &t, KeyExtractor &) {
        return t.first;
    }

    template <typename KeyExtractor>
    static auto GetKey(const Input &i, KeyExtractor & key_extractor) {
        return key_extractor(i);
    }

    template <typename FoldFunction, class T>
    static auto Fold(T&& acc, const Input &i,
                       FoldFunction & fold) {
        return TableItem(acc.first, fold(std::forward<decltype(std::forward<T>(acc).second)>(acc.second), i));
    }

    template <typename Emitter>
    static void Put(const TableItem& t, Emitter& emit) {
        emit(t);
    }
};

//! Emitter implementation to plug into a reduce hash table for
//! collecting/flushing items while reducing. Items flushed in the post-phase
//! are passed to the next DIA node for processing.
template <
    typename TableItem, typename ValueIn, typename Emitter>
class FoldPostPhaseEmitter
{
public:
    explicit FoldPostPhaseEmitter(const Emitter& emit)
        : emit_(emit) { }

    //! output an element into a partition, template specialized for VolatileKey
    //! and non-VolatileKey types
    void Emit(const TableItem& p) {
        FoldMakeTableItem<typename TableItem::first_type, ValueIn, typename TableItem::second_type>::Put(p, emit_);
    }

    //! output an element into a partition, template specialized for VolatileKey
    //! and non-VolatileKey types
    void Emit(const size_t& /* partition_id */, const TableItem& p) {
        Emit(p);
    }

public:
    //! Set of emitters, one per partition.
    Emitter emit_;
};

} // namespace core
} // namespace thrill

#endif // !THRILL_CORE_REDUCE_FUNCTIONAL_HEADER

/******************************************************************************/
