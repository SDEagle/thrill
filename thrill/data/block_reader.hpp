/*******************************************************************************
 * thrill/data/block_reader.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2015 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_DATA_BLOCK_READER_HEADER
#define THRILL_DATA_BLOCK_READER_HEADER

#include <thrill/common/config.hpp>
#include <thrill/common/item_serialization_tools.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/data/block.hpp>
#include <thrill/data/serialization.hpp>

#include <tlx/define.hpp>
#include <tlx/die.hpp>
#include <tlx/string/hexdump.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace thrill {
namespace data {

//! \addtogroup data_layer
//! \{

/*!
 * BlockReader takes Block objects from BlockSource and allows reading of
 * a) serializable Items or b) arbitray data from the Block sequence. It takes
 * care of fetching the next Block when the previous one underruns and also of
 * data items split between two Blocks.
 */
template <typename BlockSource>
class BlockReader
    : public common::ItemReaderToolsBase<BlockReader<BlockSource> >
{
public:
    static constexpr bool self_verify = common::g_self_verify;

    //! Start reading a BlockSource
    explicit BlockReader(BlockSource&& source)
        : source_(std::move(source)) { }

    //! default constructor
    BlockReader() = default;

    //! Return reference to enclosed BlockSource
    BlockSource& source() { return source_; }

    //! non-copyable: delete copy-constructor
    BlockReader(const BlockReader&) = delete;
    //! non-copyable: delete assignment operator
    BlockReader& operator = (const BlockReader&) = delete;

    //! move-constructor: default
    BlockReader(BlockReader&&) = default;
    //! move-assignment operator: default
    BlockReader& operator = (BlockReader&&) = default;

    //! return current block for debugging
    PinnedBlock CopyBlock() const {
        if (!block_.byte_block()) return PinnedBlock();
        return PinnedBlock(
            block_.CopyPinnedByteBlock(),
            current_ - block_.data_begin(), end_ - block_.data_begin(),
            block_.first_item_absolute(), num_items_, typecode_verify_);
    }

    //! return current ByteBlock
    ByteBlockPtr byte_block() const { return block_.byte_block(); }

    //! Returns typecode_verify_
    size_t typecode_verify() const { return typecode_verify_; }

    //! \name Reading (Generic) Items
    //! \{

    //! Next() reads a complete item T
    template <typename T>
    TLX_ATTRIBUTE_ALWAYS_INLINE
    T Next() {
        assert(HasNext());
        assert(num_items_ > 0);
        --num_items_;

        if (self_verify && typecode_verify_) {
            // for self-verification, T is prefixed with its hash code
            size_t code = GetRaw<size_t>();
            if (code != typeid(T).hash_code()) {
                die("BlockReader::Next() attempted to retrieve item "
                    "with different typeid! - expected "
                    << tlx::hexdump_type(typeid(T).hash_code())
                    << " got " << tlx::hexdump_type(code));
            }
        }
        return Serialization<BlockReader, T>::Deserialize(*this);
    }

    //! Next() reads a complete item T, without item counter or self
    //! verification
    template <typename T>
    TLX_ATTRIBUTE_ALWAYS_INLINE
    T NextNoSelfVerify() {
        assert(HasNext());
        return Serialization<BlockReader, T>::Deserialize(*this);
    }

    //! HasNext() returns true if at least one more item is available.
    TLX_ATTRIBUTE_ALWAYS_INLINE
    bool HasNext() {
        while (current_ == end_) {
            if (!NextBlock()) {
                return false;
            }
        }
        return true;
    }

    //! Return complete contents until empty as a std::vector<T>. Use this only
    //! if you are sure that it will fit into memory, -> only use it for tests.
    template <typename ItemType>
    std::vector<ItemType> ReadComplete() {
        std::vector<ItemType> out;
        while (HasNext()) out.emplace_back(Next<ItemType>());
        return out;
    }

    //! Read n items, however, do not deserialize them but deliver them as a
    //! vector of (unpinned) Block objects. This is used to take out a range of
    //! items, the internal item cursor is advanced by n.
    template <typename ItemType>
    std::vector<Block> GetItemBatch(size_t n) {
        static constexpr bool debug = false;

        std::vector<Block> out;
        if (n == 0) return out;

        die_unless(HasNext());
        assert(block_.IsValid());

        const Byte* begin_output = current_;
        size_t first_output = current_ - byte_block()->begin();

        // inside the if-clause the current_ may not point to a valid item
        // boundary.
        if (n >= num_items_)
        {
            // *** if the current block still contains items, push it partially

            if (n >= num_items_) {
                // construct first Block using current_ pointer
                out.emplace_back(
                    byte_block(),
                    // valid range: excludes preceding items.
                    current_ - byte_block()->begin(),
                    end_ - byte_block()->begin(),
                    // first item is at begin_ (we may have dropped some)
                    current_ - byte_block()->begin(),
                    // remaining items in this block
                    num_items_,
                    // typecode verify flag
                    typecode_verify_);

                sLOG << "partial first:" << out.back();

                n -= num_items_;

                // get next block. if not possible -> may be okay since last
                // item might just terminate the current block.
                if (!NextBlock()) {
                    assert(n == 0);
                    sLOG << "exit1 after batch.";
                    return out;
                }
            }

            // *** then append complete blocks without deserializing them

            while (n >= num_items_) {
                out.emplace_back(
                    byte_block(),
                    // full range is valid.
                    current_ - byte_block()->begin(),
                    end_ - byte_block()->begin(),
                    block_.first_item_absolute(), num_items_,
                    // typecode verify flag
                    typecode_verify_);

                sLOG << "middle:" << out.back();

                n -= num_items_;

                if (!NextBlock()) {
                    assert(n == 0);
                    sLOG << "exit2 after batch.";
                    return out;
                }
            }

            // move current_ to the first valid item of the block we got (at
            // least one NextBlock() has been called). But when constructing the
            // last Block, we have to include the partial item in the
            // front.
            begin_output = current_;
            first_output = block_.first_item_absolute();

            current_ = byte_block()->begin() + block_.first_item_absolute();
        }

        // put prospective last block into vector.

        out.emplace_back(
            byte_block(),
            // full range is valid.
            begin_output - byte_block()->begin(), end_ - byte_block()->begin(),
            first_output, n,
            // typecode verify flag
            typecode_verify_);

        // skip over remaining items in this block, there while collect all
        // blocks needed for those items via block_collect_. There can be more
        // than one block necessary for Next if an item is large!

        std::vector<PinnedBlock> out_pinned;

        block_collect_ = &out_pinned;
        if (Serialization<BlockReader, ItemType>::is_fixed_size) {
            Skip(n, n * ((self_verify && typecode_verify_ ? sizeof(size_t) : 0) +
                         Serialization<BlockReader, ItemType>::fixed_size));
        }
        else {
            while (n > 0) {
                Next<ItemType>();
                --n;
            }
        }
        block_collect_ = nullptr;

        for (PinnedBlock& pb : out_pinned)
            out.emplace_back(std::move(pb).MoveToBlock());
        out_pinned.clear();

        out.back().set_end(current_ - byte_block()->begin());

        sLOG << "partial last:" << out.back();

        sLOG << "exit3 after batch:"
             << "current_=" << current_ - byte_block()->begin();

        return out;
    }

    //! \}

    //! \name Cursor Reading Methods
    //! \{

    //! Fetch a number of unstructured bytes from the current block, advancing
    //! the cursor.
    BlockReader& Read(void* outdata, size_t size) {

        Byte* cdata = reinterpret_cast<Byte*>(outdata);

        while (TLX_UNLIKELY(current_ + size > end_)) {
            // partial copy of remainder of block
            size_t partial_size = end_ - current_;
            std::copy(current_, current_ + partial_size, cdata);

            cdata += partial_size;
            size -= partial_size;

            if (!NextBlock())
                throw std::runtime_error("Data underflow in BlockReader.");
        }

        // copy rest from current block
        std::copy(current_, current_ + size, cdata);
        current_ += size;

        return *this;
    }

    //! Fetch a number of unstructured bytes from the buffer as std::string,
    //! advancing the cursor.
    std::string Read(size_t datalen) {
        std::string out(datalen, 0);
        Read(const_cast<char*>(out.data()), out.size());
        return out;
    }

    //! Advance the cursor given number of bytes without reading them.
    BlockReader& Skip(size_t items, size_t bytes) {
        while (TLX_UNLIKELY(current_ + bytes > end_)) {
            bytes -= end_ - current_;
            // deduct number of remaining items in skipped block from item skip
            // counter.
            items -= num_items_;
            if (!NextBlock())
                throw std::runtime_error("Data underflow in BlockReader.");
        }
        current_ += bytes;
        // the last line skipped over the remaining "items" number of items.
        num_items_ -= items;
        return *this;
    }

    //! Fetch a single byte from the current block, advancing the cursor.
    Byte GetByte() {
        // loop, since blocks can actually be empty.
        while (TLX_UNLIKELY(current_ == end_)) {
            if (!NextBlock())
                throw std::runtime_error("Data underflow in BlockReader.");
        }
        return *current_++;
    }

    //! Fetch a single item of the template type Type from the buffer,
    //! advancing the cursor. Be careful with implicit type conversions!
    template <typename Type>
    Type GetRaw() {
        static_assert(std::is_pod<Type>::value,
                      "You only want to GetRaw() POD types as raw values.");

        Type ret;

        // fast path for reading item from block if it fits.
        if (TLX_LIKELY(current_ + sizeof(Type) <= end_)) {
            ret = *reinterpret_cast<const Type*>(current_);
            current_ += sizeof(Type);
        }
        else {
            Read(&ret, sizeof(ret));
        }

        return ret;
    }

    //! \}

private:
    //! Instance of BlockSource. This is NOT a reference, as to enable embedding
    //! of FileBlockSource to compose classes into File::Reader.
    BlockSource source_;

    //! The current block being read, this holds a shared pointer reference.
    PinnedBlock block_;

    //! current read pointer into current block of file.
    const Byte* current_ = nullptr;

    //! pointer to end of current block.
    const Byte* end_ = nullptr;

    //! remaining number of items starting in this block
    size_t num_items_ = 0;

    //! pointer to vector to collect blocks in GetItemRange.
    std::vector<PinnedBlock>* block_collect_ = nullptr;

    //! flag whether the underlying data contains self verify type codes from
    //! BlockReader, this is false to needed to read external files.
    bool typecode_verify_;

    //! Call source_.NextBlock with appropriate parameters
    bool NextBlock() {
        // first release old pin.
        block_.Reset();
        // request next pinned block
        block_ = source_.NextBlock();
        sLOG0 << "BlockReader::NextBlock" << block_;

        if (!block_.IsValid()) return false;

        if (block_collect_)
            block_collect_->emplace_back(block_);

        current_ = block_.data_begin();
        end_ = block_.data_end();
        num_items_ = block_.num_items();
        typecode_verify_ = block_.typecode_verify();
        return true;
    }
};

//! \}

} // namespace data
} // namespace thrill

#endif // !THRILL_DATA_BLOCK_READER_HEADER

/******************************************************************************/
