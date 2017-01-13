/*******************************************************************************
 * thrill/core/duplicate_detection.hpp
 *
 * Duplicate detection
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2016 Alexander Noe <aleexnoe@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_CORE_DUPLICATE_DETECTION_HEADER
#define THRILL_CORE_DUPLICATE_DETECTION_HEADER

#include <thrill/api/context.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/core/dynamic_bitset.hpp>
#include <thrill/core/golomb_reader.hpp>
#include <thrill/core/multiway_merge.hpp>

#include <algorithm>
#include <memory>

namespace thrill {
namespace core {

/*!
 * Duplicate detection to identify all elements occuring only on one worker.
 * This information can be used to locally reduce uniquely-occuring elements.
 * Therefore this saves communication volume in operations such as api::Reduce()
 * or api::Join().
 *
 * Internally, this duplicate detection uses a golomb encoded distributed single
 * shot bloom filter to find duplicates and non-duplicates with as low
 * communication volume as possible. Due to the bloom filter's inherent
 * properties, this has false duplicates but no false non-duplicates.
 *
 * Should only be used when a large amount of uniquely-occuring elements are
 * expected.
 */
class DuplicateDetection
{

private:
    /*!
     * Sends all hashes in the range
     * [max_hash / num_workers * p, max_hash / num_workers * (p + 1)) to worker
     * p. These hashes are encoded with a golomb encoder in core::DynamicBitset.
     *
     * \param stream_pointer Pointer to data stream
     * \param hashes Sorted vector of all hashes modulo max_hash
     * \param b Golomb parameter
     * \param space_bound Upper bound of used space for golomb codes
     * \param num_workers Number of workers in this Thrill process
     * \param max_hash Modulo for all hashes
     */
    void WriteEncodedHashes(data::CatStreamPtr stream_pointer,
                            const std::vector<size_t>& hashes,
                            size_t b,
                            size_t space_bound,
                            size_t num_workers,
                            size_t max_hash) {

        std::vector<data::CatStream::Writer> writers =
            stream_pointer->GetWriters();

        size_t j = 0;
        for (size_t i = 0; i < num_workers; ++i) {
            common::Range range_i = common::CalculateLocalRange(max_hash, num_workers, i);

            // TODO: Lower bound.
            core::DynamicBitset<size_t>
            golomb_code(space_bound, false, b);

            golomb_code.seek();

            size_t delta = 0;
            size_t num_elements = 0;

            //! Local duplicates are only sent once, this is detected by checking equivalence
            //! to previous element. Thus we need a special case for the first element being 0
            if (j < hashes.size() && hashes[j] == 0) {
                golomb_code.golomb_in(0);
                ++num_elements;
                ++j;
            }

            for (            /*j is already set from previous workers*/
                ; j < hashes.size() && hashes[j] < range_i.end; ++j) {
                //! Send hash deltas to make the encoded bitset smaller.
                if (hashes[j] != delta) {
                    ++num_elements;
                    golomb_code.golomb_in(hashes[j] - delta);
                    delta = hashes[j];
                }
            }

            //! Send raw data through data stream.
            writers[i].Put(golomb_code.size());
            writers[i].Put(num_elements);
            writers[i].Append(golomb_code.GetGolombData(),
                              golomb_code.size() * sizeof(size_t));
            writers[i].Close();
        }
    }

    /*!
     * Reads a golomb encoded bitset from a data stream and returns it's contents
     * in form of a vector of hashes.
     *
     * \param stream_pointer Pointer to data stream
     * \param target_vec Target vector for hashes, should be empty beforehand
     * \param b Golomb parameter
     */
    void ReadEncodedHashesToVector(data::CatStreamPtr stream_pointer,
                                   std::vector<bool>& target_vec,
                                   size_t b) {

        auto reader = stream_pointer->GetCatReader(/* consume */ true);

        while (reader.HasNext()) {

            size_t data_size = reader.template Next<size_t>();
            size_t num_elements = reader.template Next<size_t>();
            if (num_elements) {
                size_t* raw_data = new size_t[data_size];
                reader.Read(raw_data, data_size * sizeof(size_t));

                //! Builds golomb encoded bitset from data recieved by the stream.
                core::DynamicBitset<size_t> golomb_code(raw_data,
                                                        data_size,
                                                        b, num_elements);
                golomb_code.seek();

                size_t last = 0;
                for (size_t i = 0; i < num_elements; ++i) {
                    //! Golomb code contains deltas, we want the actual values
                    size_t new_elem = golomb_code.golomb_out() + last;
                    target_vec[new_elem] = true;

                    last = new_elem;
                }

                delete[] raw_data;
            }
        }
    }

public:
    /*!
     * Identifies all hashes which occur on only a single worker.
     * Returns all local uniques in form of a vector of hashes.
     *
     * \param non_duplicates Empty vector, which contains all non-duplicate
     * hashes after this method
     * \param hashes Hashes for all elements on this worker.
     * \param context Thrill context, used for collective communication
     * \param dia_id Id of the operation, which calls this method. Used
     *   to uniquely identify the data streams used.
     *
     * \return Modulo used on all hashes. (Use this modulo on all hashes to
     *  identify possible non-duplicates)
     */
    size_t FindNonDuplicates(std::vector<bool>& non_duplicates,
                             std::vector<size_t>& hashes,
                             Context& context,
                             size_t dia_id) {

        //! This bound could often be lowered when we have many duplicates.
        //! This would however require a large amount of added communication.
        size_t upper_bound_uniques = context.net.AllReduce(hashes.size());

        //! Golomb Parameters taken from original paper (Sanders, Schlag, Müller)

        //! Parameter for false positive rate (FPR: 1/fpr_parameter)
        double fpr_parameter = 8;
        size_t b = (size_t)fpr_parameter;  //(size_t)(std::log(2) * fpr_parameter);
        size_t upper_space_bound = upper_bound_uniques *
                                   (2 + std::log2(fpr_parameter));
        size_t max_hash = upper_bound_uniques * fpr_parameter;

        for (size_t i = 0; i < hashes.size(); ++i) {
            hashes[i] = hashes[i] % max_hash;
        }

        std::sort(hashes.begin(), hashes.end());

        data::CatStreamPtr golomb_data_stream = context.GetNewCatStream(dia_id);

        WriteEncodedHashes(golomb_data_stream,
                           hashes, b,
                           upper_space_bound,
                           context.num_workers(),
                           max_hash);

        std::vector<data::BlockReader<data::ConsumeBlockQueueSource> > readers =
            golomb_data_stream->GetReaders();

        std::vector<GolombReader> g_readers;
        std::vector<std::unique_ptr<size_t[]> > data_pointers;

        data_pointers.reserve(context.num_workers());

        size_t total_elements = 0;

        for (auto& reader : readers) {
            assert(reader.HasNext());
            size_t data_size = reader.template Next<size_t>();
            size_t num_elements = reader.template Next<size_t>();
            data_pointers.push_back(
                std::make_unique<size_t[]>(data_size + 1));

            reader.Read(data_pointers.back().get(), data_size * sizeof(size_t));

            total_elements += num_elements;

            g_readers.push_back(
                GolombReader(data_size, data_pointers.back().get(), num_elements, b));
        }

        auto puller = make_multiway_merge_tree<size_t>
                          (g_readers.begin(), g_readers.end(),
                          [](const size_t& hash1,
                             const size_t& hash2) {
                              return hash1 < hash2;
                          });

        data::CatStreamPtr duplicates_stream = context.GetNewCatStream(dia_id);

        std::vector<data::CatStream::Writer> duplicate_writers =
            duplicates_stream->GetWriters();

        std::vector<core::DynamicBitset<size_t>*> bitsets;
        std::vector<size_t> deltas(context.num_workers(), 0);
        std::vector<size_t> element_counters(context.num_workers(), 0);

        for (size_t i = 0; i < context.num_workers(); ++i) {
            bitsets.emplace_back(new core::DynamicBitset<size_t>(
                                     upper_space_bound, false, b));
        }

        std::pair<size_t, size_t> this_element;
        std::pair<size_t, size_t> next_element;

        size_t ctr = 0;

        if (total_elements > 0) {
            next_element = puller.NextWithSource();

            while (ctr < total_elements - 1) {

                this_element = next_element;
                next_element = puller.NextWithSource();
                ctr++;
                //! find all keys only occuring on a single worker and insert
                //! to according bitset

                if (this_element.first != next_element.first) {
                    size_t proc = this_element.second;
                    bitsets[proc]->golomb_in(this_element.first -
                                             deltas[proc]);
                    deltas[proc] = this_element.first;
                    element_counters[proc]++;
                } else {
                    size_t cmp = next_element.first;
                    while (puller.HasNext() &&
                           next_element.first == cmp) {
                        next_element = puller.NextWithSource();
                        ctr++;
                    }
                }
            }
        }

        if (this_element.first != next_element.first) {
            bitsets[next_element.second]->golomb_in(next_element.first -
                                                    deltas[next_element.second]);
            element_counters[next_element.second]++;
        }

        assert(!puller.HasNext());

        for (size_t i = 0; i < context.num_workers(); ++i) {
            duplicate_writers[i].Put(bitsets[i]->size());
            duplicate_writers[i].Put(element_counters[i]);
            duplicate_writers[i].Append(bitsets[i]->GetGolombData(),
                                        bitsets[i]->size() * sizeof(size_t));
            duplicate_writers[i].Close();
            delete bitsets[i];
        }

        assert(!non_duplicates.size());
        non_duplicates.resize(max_hash);
        ReadEncodedHashesToVector(duplicates_stream,
                                  non_duplicates, b);

        return max_hash;
    }
};

} // namespace core
} // namespace thrill

#endif // !THRILL_CORE_DUPLICATE_DETECTION_HEADER

/******************************************************************************/
