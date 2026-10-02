import torch

import training


def test_stratified_smoke_subset_keeps_both_classes():
    index = [(0, i, 0) for i in range(10)] + [(1, i, 2) for i in range(20)]
    result = training.stratified_smoke_subset(index, 8)
    assert len(result) == 8
    assert {row[2] for row in result} == {0, 2}


def test_balanced_accuracy_uses_fixed_half_threshold():
    result = training.balanced_accuracy(
        training.np.asarray([0, 0, 1, 1]),
        training.np.asarray([0.1, 0.6, 0.4, 0.9]),
    )
    assert result["balanced_accuracy"] == 0.5
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]


def test_slip_branch_and_continuation_are_exact():
    training.configure_determinism()
    training.seed_all(11)
    branch = training.SlipBranch(training.p2.build_decoder("decoupled"))
    tokens = torch.randn(2, 9, 768)
    labels = torch.tensor([0, 1])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(branch.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    loss = criterion(branch(tokens), labels)
    loss.backward()
    optimizer.step()
    serialized = training.checkpoint_payload(branch, optimizer, 1, [], 1, 0.5, 0, {}, {})
    proof = training.same_next_step_proof(
        branch, serialized, criterion, (tokens, labels, None, None, None), torch.device("cpu")
    )
    assert proof["pass"] is True
