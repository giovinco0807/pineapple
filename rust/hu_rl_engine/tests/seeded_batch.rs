use ofc_hu_rl_engine::{
    paired_seeded_decks, BatchExecutionConfig, BatchHuRlEnv, BatchStepOutcomeV1, PairedSeedRange,
    ScalarHuRlEnv, DECISION_COUNT,
};

#[test]
fn seeded_explicit_and_scalar_paths_are_byte_exact_for_all_ten_decisions() {
    let range = PairedSeedRange::new(12_345, 7, 3, 17).unwrap();
    let decks = paired_seeded_decks(range);
    let mut seeded = BatchHuRlEnv::with_execution_config_from_paired_seed_range(
        range,
        BatchExecutionConfig::new(2, 4).unwrap(),
    )
    .unwrap();
    let mut explicit =
        BatchHuRlEnv::with_execution_config(&decks, BatchExecutionConfig::new(5, 2).unwrap())
            .unwrap();
    let mut scalars = decks
        .iter()
        .map(|deck| ScalarHuRlEnv::new(deck).unwrap())
        .collect::<Vec<_>>();

    for decision in 0..DECISION_COUNT {
        let scalar_views = scalars
            .iter()
            .map(|lane| lane.observe().unwrap())
            .collect::<Vec<_>>();
        let scalar_mappings = scalars
            .iter()
            .map(|lane| lane.legal_mapping().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(seeded.observe_batch().unwrap(), scalar_views);
        assert_eq!(explicit.observe_batch().unwrap(), scalar_views);
        assert_eq!(seeded.legal_mappings_batch().unwrap(), scalar_mappings);
        assert_eq!(explicit.legal_mappings_batch().unwrap(), scalar_mappings);

        let selected = scalar_mappings
            .iter()
            .enumerate()
            .map(|(lane, mapping)| {
                let index = (decision * 17 + lane * 11) % mapping.action_count();
                mapping.key_at(index).unwrap()
            })
            .collect::<Vec<_>>();
        let scalar_results = scalars
            .iter_mut()
            .zip(selected.iter().copied())
            .map(|(lane, key)| BatchStepOutcomeV1::from(lane.step(key).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(seeded.step_batch(&selected).unwrap(), scalar_results);
        assert_eq!(explicit.step_batch(&selected).unwrap(), scalar_results);
        assert_eq!(seeded.decision_counts(), explicit.decision_counts());
    }

    assert!(seeded.all_done());
    assert!(explicit.all_done());
    assert_eq!(
        seeded.terminal_rewards_batch().unwrap(),
        explicit.terminal_rewards_batch().unwrap()
    );
}

#[test]
fn paired_reset_is_atomic_and_success_advances_snapshot_lineage() {
    let initial = PairedSeedRange::new(41, 2, 2, 3).unwrap();
    let mut batch = BatchHuRlEnv::from_paired_seed_range(initial).unwrap();
    let selected = batch
        .legal_mappings_batch()
        .unwrap()
        .iter()
        .map(|mapping| mapping.key_at(0).unwrap())
        .collect::<Vec<_>>();
    batch.step_batch(&selected).unwrap();
    let before_rejected = batch.snapshot();
    let before_views = batch.observe_batch().unwrap();

    let wrong_width = PairedSeedRange::new(99, 0, 1, 1).unwrap();
    assert!(batch.reset_from_paired_seed_range(wrong_width).is_err());
    assert_eq!(batch.snapshot(), before_rejected);
    assert_eq!(batch.observe_batch().unwrap(), before_views);
    // A rejected reset did not advance lineage; the existing snapshot remains usable.
    batch.restore(&before_rejected).unwrap();

    let replacement = PairedSeedRange::new(73, 5, 2, 19).unwrap();
    let expected = paired_seeded_decks(replacement)
        .iter()
        .map(|deck| ScalarHuRlEnv::new(deck).unwrap().observe().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        batch.reset_from_paired_seed_range(replacement).unwrap(),
        expected
    );
    assert_eq!(batch.observe_batch().unwrap(), expected);
    assert_eq!(batch.decision_counts(), vec![0; 4]);
    assert!(batch.restore(&before_rejected).is_err());
    assert_eq!(batch.observe_batch().unwrap(), expected);
}
