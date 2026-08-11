use ofc_hu_rl_engine::{
    ActionKey, BatchExecutionConfig, BatchHuRlEnv, BatchStepOutcomeV1, Card, LegalActionMappingV1,
    ScalarHuRlEnv, ALL_CARDS, DECISION_COUNT, MAX_BATCH_LANES, MAX_BATCH_THREADS,
    RAW_STEP_RESULT_ARTIFACT_ROLE,
};

fn decks() -> Vec<Vec<Card>> {
    [0, 5, 17, 31]
        .into_iter()
        .enumerate()
        .map(|(lane, offset)| {
            let mut deck = ALL_CARDS.to_vec();
            deck.rotate_left(offset);
            if lane % 2 == 1 {
                deck.reverse();
            }
            deck
        })
        .collect()
}

fn selected_key(mapping: &LegalActionMappingV1, lane: usize, decision: usize) -> ActionKey {
    let action_count = mapping.action_count();
    let index = match (lane + decision) % 3 {
        0 => 0,
        1 => action_count / 2,
        _ => action_count - 1,
    };
    mapping.key_at(index).unwrap()
}

fn selections(batch: &BatchHuRlEnv, decision: usize) -> Vec<ActionKey> {
    batch
        .legal_mappings_batch()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(lane, mapping)| selected_key(mapping, lane, decision))
        .collect()
}

#[test]
fn batch_is_exactly_scalar_and_invariant_to_chunk_and_thread_hints() {
    let decks = decks();
    let mut scalars = decks
        .iter()
        .map(|deck| ScalarHuRlEnv::new(deck).unwrap())
        .collect::<Vec<_>>();
    let configs = [(1, 1), (2, 4), (3, 64), (97, 2)]
        .map(|(chunk, threads)| BatchExecutionConfig::new(chunk, threads).unwrap());
    let mut batches = configs
        .into_iter()
        .map(|config| BatchHuRlEnv::with_execution_config(&decks, config).unwrap())
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
        for batch in &batches {
            assert_eq!(batch.observe_batch().unwrap(), scalar_views);
            assert_eq!(batch.legal_mappings_batch().unwrap(), scalar_mappings);
            assert_eq!(batch.decision_counts(), vec![decision; decks.len()]);
        }

        let selected = scalar_mappings
            .iter()
            .enumerate()
            .map(|(lane, mapping)| selected_key(mapping, lane, decision))
            .collect::<Vec<_>>();
        let scalar_results = scalars
            .iter_mut()
            .zip(selected.iter().copied())
            .map(|(lane, key)| BatchStepOutcomeV1::from(lane.step(key).unwrap()))
            .collect::<Vec<_>>();
        for batch in &mut batches {
            assert_eq!(batch.step_batch(&selected).unwrap(), scalar_results);
        }
    }

    let scalar_rewards = scalars
        .iter()
        .map(|lane| lane.terminal_rewards().unwrap())
        .collect::<Vec<_>>();
    for batch in &batches {
        assert!(batch.all_done());
        assert_eq!(batch.terminal_rewards_batch().unwrap(), scalar_rewards);
        assert_eq!(batch.snapshot().decision_counts(), vec![10; decks.len()]);
    }
}

#[test]
fn one_invalid_lane_rolls_back_the_entire_step() {
    let decks = decks();
    let mut batch =
        BatchHuRlEnv::with_execution_config(&decks, BatchExecutionConfig::new(1, 4).unwrap())
            .unwrap();
    let before = batch.snapshot();
    let mappings = batch.legal_mappings_batch().unwrap();
    let selected = vec![
        mappings[0].key_at(0).unwrap(),
        ActionKey::default(),
        mappings[2].key_at(mappings[2].action_count() - 1).unwrap(),
        ActionKey::default(),
    ];
    let error = batch.step_batch(&selected).unwrap_err();
    assert!(error.message().contains("lane 1"));
    assert!(error.message().contains("not legal"));
    assert_eq!(batch.snapshot(), before);

    let error = batch.step_batch(&selected[..3]).unwrap_err();
    assert!(error.message().contains("exactly 4 lanes"));
    assert_eq!(batch.snapshot(), before);
}

#[test]
fn lane_order_is_preserved_by_observe_step_and_atomic_reset() {
    let original = decks();
    let mut batch =
        BatchHuRlEnv::with_execution_config(&original, BatchExecutionConfig::new(3, 8).unwrap())
            .unwrap();
    let expected_initial = original
        .iter()
        .map(|deck| {
            ScalarHuRlEnv::new(deck)
                .unwrap()
                .observe()
                .unwrap()
                .digest()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        batch
            .observe_batch()
            .unwrap()
            .iter()
            .map(|view| view.digest())
            .collect::<Vec<_>>(),
        expected_initial
    );

    let first_actions = selections(&batch, 0);
    let first_results = batch.step_batch(&first_actions).unwrap();
    for (result, selected) in first_results.iter().zip(&first_actions) {
        assert_eq!(
            result.public_placement.placement_masks(),
            [
                selected.top_mask,
                selected.middle_mask,
                selected.bottom_mask
            ]
        );
        assert_eq!(
            result.public_placement.discard_count() as u32,
            selected.discard_mask.count_ones()
        );
    }

    let mut reordered = original.clone();
    reordered.rotate_left(1);
    let reset_views = batch.reset_from_explicit_decks(&reordered).unwrap();
    let expected_reset = reordered
        .iter()
        .map(|deck| ScalarHuRlEnv::new(deck).unwrap().observe().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(reset_views, expected_reset);
    assert_eq!(batch.observe_batch().unwrap(), expected_reset);
    assert_eq!(batch.decision_counts(), vec![0; reordered.len()]);

    let before_invalid_reset = batch.snapshot();
    let mut invalid = reordered.clone();
    invalid[2][51] = invalid[2][0];
    let error = batch.reset_from_explicit_decks(&invalid).unwrap_err();
    assert!(error.message().contains("lane 2"));
    assert!(error.message().contains("duplicate"));
    assert_eq!(batch.snapshot(), before_invalid_reset);
}

#[test]
fn snapshot_restore_replays_all_lanes_and_excludes_execution_hints() {
    let decks = decks();
    let mut batch =
        BatchHuRlEnv::with_execution_config(&decks, BatchExecutionConfig::new(2, 2).unwrap())
            .unwrap();
    for decision in 0..4 {
        let selected = selections(&batch, decision);
        batch.step_batch(&selected).unwrap();
    }
    let checkpoint = batch.snapshot();
    let checkpoint_views = batch.observe_batch().unwrap();
    assert_eq!(checkpoint.decision_counts(), vec![4; decks.len()]);
    let debug = format!("{checkpoint:?}");
    assert!(debug.contains("<redacted>"));
    assert!(!debug.contains("As"));

    let mut replay = Vec::<(Vec<ActionKey>, Vec<BatchStepOutcomeV1>)>::new();
    for decision in 4..DECISION_COUNT {
        let selected = selections(&batch, decision);
        let results = batch.step_batch(&selected).unwrap();
        replay.push((selected, results));
    }
    let expected_terminal = batch.snapshot();
    let expected_rewards = batch.terminal_rewards_batch().unwrap();

    let changed_hints = BatchExecutionConfig::new(1, 64).unwrap();
    batch.set_execution_config(changed_hints).unwrap();
    batch.restore(&checkpoint).unwrap();
    assert_eq!(batch.execution_config(), changed_hints);
    assert_eq!(batch.snapshot(), checkpoint);
    assert_eq!(batch.observe_batch().unwrap(), checkpoint_views);
    for (selected, expected_results) in replay {
        assert_eq!(batch.step_batch(&selected).unwrap(), expected_results);
    }
    assert_eq!(batch.snapshot(), expected_terminal);
    assert_eq!(batch.terminal_rewards_batch().unwrap(), expected_rewards);
}

#[test]
fn snapshots_are_bound_to_batch_and_reset_generation_lineage() {
    let decks = decks();
    let mut first = BatchHuRlEnv::from_explicit_decks(&decks).unwrap();
    let second = BatchHuRlEnv::from_explicit_decks(&decks).unwrap();
    let mut reordered = decks.clone();
    reordered.rotate_left(1);
    let reordered_batch = BatchHuRlEnv::from_explicit_decks(&reordered).unwrap();

    let original = first.snapshot();
    for foreign in [second.snapshot(), reordered_batch.snapshot()] {
        let before = first.snapshot();
        let error = first.restore(&foreign).unwrap_err();
        assert!(error.message().contains("lineage"));
        assert!(!error.message().contains("batch_id"));
        assert_eq!(first.snapshot(), before);
    }

    first.reset_from_explicit_decks(&reordered).unwrap();
    let after_reset = first.snapshot();
    let error = first.restore(&original).unwrap_err();
    assert!(error.message().contains("lineage"));
    assert_eq!(first.snapshot(), after_reset);
    let debug = format!("{after_reset:?}");
    assert!(!debug.contains("lineage"));
    assert!(!debug.contains("batch_id"));
}

#[test]
fn invalid_batch_shape_and_execution_hints_fail_closed() {
    assert!(BatchHuRlEnv::from_explicit_decks(&[])
        .unwrap_err()
        .message()
        .contains("at least one lane"));
    assert!(BatchExecutionConfig::new(0, 1)
        .unwrap_err()
        .message()
        .contains("chunk_width"));
    assert!(BatchExecutionConfig::new(1, 0)
        .unwrap_err()
        .message()
        .contains("thread_count"));
    assert!(BatchExecutionConfig::new(1, MAX_BATCH_THREADS + 1)
        .unwrap_err()
        .message()
        .contains("maximum 64"));

    let over_limit = vec![ALL_CARDS.to_vec(); MAX_BATCH_LANES + 1];
    assert!(BatchHuRlEnv::from_explicit_decks(&over_limit)
        .unwrap_err()
        .message()
        .contains("maximum 4096"));
}

#[test]
fn raw_step_debug_is_privileged_and_batch_outcome_erases_action_key() {
    let mut scalar = ScalarHuRlEnv::new(&ALL_CARDS).unwrap();
    for _ in 0..2 {
        let mapping = scalar.legal_mapping().unwrap();
        scalar
            .step(mapping.key_at(mapping.action_count() / 2).unwrap())
            .unwrap();
    }
    let mapping = scalar.legal_mapping().unwrap();
    let selected = mapping.key_at(mapping.action_count() / 2).unwrap();
    let discard = selected.cards("discards").unwrap()[0];
    let raw = scalar.step(selected).unwrap();
    let raw_debug = format!("{raw:?}");
    assert!(raw_debug.contains(RAW_STEP_RESULT_ARTIFACT_ROLE));
    assert!(raw_debug.contains("<redacted privileged ActionKey>"));
    assert!(!raw_debug.contains(&selected.to_token()));
    assert!(!raw_debug.contains(discard.as_str()));
    assert!(!raw.policy_input_eligible());
    assert!(!raw.replay_eligible());
    assert!(!raw.training_eligible());

    let safe = BatchStepOutcomeV1::from(raw);
    let safe_debug = format!("{safe:?}");
    assert!(!safe_debug.contains("action_key"));
    assert!(!safe_debug.contains(&selected.to_token()));
    assert!(!safe_debug.contains(discard.as_str()));
    assert_eq!(safe.public_placement.discard_count(), 1);
}
