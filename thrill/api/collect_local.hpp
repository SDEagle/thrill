/*******************************************************************************
 * thrill/api/collect_local.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2015 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_COLLECT_LOCAL_HEADER
#define THRILL_API_COLLECT_LOCAL_HEADER

#include <thrill/api/action_node.hpp>
#include <thrill/api/dia.hpp>

#include <iostream>
#include <vector>

namespace thrill {
namespace api {

/*!
 * \ingroup api_layer
 */
template <typename ValueType>
class CollectLocalNode final : public ActionResultNode<std::vector<ValueType> >
{
public:
    using Super = ActionResultNode<std::vector<ValueType> >;
    using Super::context_;

    template <typename ParentDIA>
    CollectLocalNode(const ParentDIA& parent, const char* label,
               std::vector<ValueType>* out_vector)
        : Super(parent.ctx(), label,
                { parent.id() }, { parent.node() }),
          out_vector_(out_vector)
    {
        auto pre_op_fn = [this](const ValueType& input) {
                             out_vector_->push_back(input);
                         };

        // close the function stack with our pre op and register it at parent
        // node for output
        auto lop_chain = parent.stack().push(pre_op_fn).fold();
        parent.node()->AddChild(this, lop_chain);
    }

    void Execute() final {}

    const std::vector<ValueType>& result() const final {
        return *out_vector_;
    }

private:
    //! Vector pointer to write elements to.
    std::vector<ValueType>* out_vector_;
};

template <typename ValueType, typename Stack>
std::vector<ValueType>
DIA<ValueType, Stack>::CollectLocal() const {
    assert(IsValid());

    using CollectLocalNode = api::CollectLocalNode<ValueType>;

    std::vector<ValueType> output;

    auto node = common::MakeCounting<CollectLocalNode>(
        *this, "CollectLocal", &output);

    node->RunScope();

    return output;
}

template <typename ValueType, typename Stack>
void DIA<ValueType, Stack>::CollectLocal(
    std::vector<ValueType>* out_vector) const {
    assert(IsValid());

    using CollectLocalNode = api::CollectLocalNode<ValueType>;

    auto node = common::MakeCounting<CollectLocalNode>(
        *this, "CollectLocal", out_vector);

    node->RunScope();
}

} // namespace api
} // namespace thrill

#endif // !THRILL_API_COLLECT_LOCAL_HEADER

/******************************************************************************/
