use ofc_hu_m3_engine::cards::ALL_CARDS;
use ofc_hu_rl_engine::{
    ActOrder, ActionKey, Card, HuRlActorViewV1, PublicPlacement, ScalarHuRlEnv, Seat, Street,
    DECISION_COUNT, DECISION_SCHEDULE, MAX_LEGAL_ACTIONS,
};
use std::collections::BTreeMap;
use std::collections::HashSet;

const PYTHON_MIDDLE_KEYS: [&str; 10] = [
    "rak1:0000000000008:0000000000010:0000000000007:0000000000000",
    "rak1:0000000000100:0000000000200:00000000000e0:0000000000000",
    "rak1:0000000000400:0000000000000:0000000001000:0000000000800",
    "rak1:0000000002000:0000000000000:0000000008000:0000000004000",
    "rak1:0000000010000:0000000000000:0000000040000:0000000020000",
    "rak1:0000000080000:0000000000000:0000000200000:0000000100000",
    "rak1:0000000000000:0000001400000:0000000000000:0000000800000",
    "rak1:0000000000000:000000a000000:0000000000000:0000004000000",
    "rak1:0000000000000:0000050000000:0000000000000:0000020000000",
    "rak1:0000000000000:0000280000000:0000000000000:0000100000000",
];

const PYTHON_VIEW_DIGESTS: [&str; 10] = [
    "9b151c2c92bade6151ac6c63c337c603ff80cdc678dc61d2e783f0c586c79ca7",
    "e93d5cd558619db9d329f17c26f3b2ba42e444d89b71e1a40d167d693f958fc0",
    "6fd0337f6e97a893d0a6a249cf62f2c887ff2505e26c83b6ab3ba683b0cf260b",
    "2a6bbd4b9db32683475796c28167842721d3b543c5599bc66df9229881167c03",
    "c2ec1ab407bf31b5f23f7ba560a619e0ce9a28be1c7ce1e2bee0a15442c97225",
    "6f7892f5c37d7bd7effa76dffd57e4ea259bc4345b4c041d8d967fc1c3881861",
    "a3a1f9efd74633911e42b5dfc334f1c58056ec587a78871fbe3044d97fe6a22f",
    "9c912b69343afe351783795d60c201f3679c0a925dd825ac059b0518ce1fed8e",
    "8b9183444eb6b8da1fa891e5f0af38dc755f7cde7ffc5be24f038081371de85f",
    "7742a3e88ad182e3d3c837bdd9fd9b32c23e5afb08cd78a6bafc1bb74f930651",
];

// Exact outputs of Python
// regular_hu_full_hand_all_cards_middle_key_v1, canonical fixture SHA-256
// 60c634aa8fb91f1ebdbf507f508a573c337f6e6838e45e8d0c8b80ab62e27e3e.
const PYTHON_GOLDEN_V1_VIEW_DIGESTS: [&str; 10] = [
    "914f2712409366eaac6cb4b590b875ebbfb1c16dbf4fb87c32dac7c092e622d7",
    "fd623bf7f64416f6c32583e008875ceffc0f604fd6c924eb76d52917442791c6",
    "56b9cc8a46c3d8953ea02d0040a4aa8479a9090eb59a446502987a0780cdb744",
    "00a2145feeb080b845245cc33070280a958b49fa50fa8e31e678017d05125381",
    "2a0a4105ce19f826f5f729fbe0b692c05a2151d884a323c4184fe818febb2211",
    "1447379e2105e794c41876052dbd6089be6fd3df7618153a0bbdc445330b50bb",
    "1b03b45578e263b4604540ee7e4652270c61b65b250ab0a68b457e83a75f55c7",
    "34637961e59d547d71977486f4827fdc69347b661685872ab1a40f4eb2f09891",
    "37c1cd413afc6989cb0d5c3920a07bf73738e57e17929fa54f13f626308a48f8",
    "e8fc47be9ddcf87279909b39ebd749b2864e8ecd3d6b38375af6a016c3ae61ca",
];

const PYTHON_GOLDEN_V1_MIDDLE_KEYS: [&str; 10] = [
    "rak1:0000000000100:0000000000200:00000000000e0:0000000000000",
    "rak1:0000000002000:0000000004000:0000000001c00:0000000000000",
    "rak1:0000000008000:0000000000000:0000000020000:0000000010000",
    "rak1:0000000040000:0000000000000:0000000100000:0000000080000",
    "rak1:0000000200000:0000000000000:0000000800000:0000000400000",
    "rak1:0000001000000:0000000000000:0000004000000:0000002000000",
    "rak1:0000000000000:0000028000000:0000000000000:0000010000000",
    "rak1:0000000000000:0000140000000:0000000000000:0000080000000",
    "rak1:0000000000000:0000a00000000:0000000000000:0000400000000",
    "rak1:0000000000000:0005000000000:0000000000000:0002000000000",
];

const PYTHON_GOLDEN_V1_ACTION_SET_DIGESTS: [&str; 10] = [
    "d9de460d654dd7c0609f14a7754e05ba7cfba01e70bb8f692cdeef2eee887520",
    "3763fa2d14ee0eba4ef00d8e79cf027586eb956cd8ffbbf659c211e777cf7f54",
    "83cca7432443417d952c3800997644c7b3015e9a349ca2b073948aebd6d81cb8",
    "f191c22656a02d78d70b3e2da6975b7077c8c335ad358d23cd400ba2a7c43c00",
    "d8f2f4773ecf403f24c322d1a3c381e5294b81aec7021496cfca2e0c2eaad76d",
    "e6eb672ff67b5e3c5b3dfc2ca426da047a4d7e99e975b80b4c6904dd9c2f3583",
    "62c29afd9b54b869b9e6eef35755ac4097c31661f4fc5e1c1a48fba11805d98e",
    "cd388ddb46903e7433abef9b0b13ebb80982a5166272634d512e948925a13a34",
    "50a2dd424111f9a9562190da7c2cdba607a78155325d5b44109c20246c76840b",
    "c244cbcb52ac92ef5cef926c671554c2a9bb22498bd1fc1094cf5f27a9e068c5",
];

const PYTHON_GOLDEN_V1_ACTION_COUNTS: [usize; 10] = [232, 232, 27, 27, 21, 21, 3, 3, 3, 3];

fn middle_key(env: &ScalarHuRlEnv) -> ActionKey {
    let keys = env.legal_actions().unwrap();
    keys[keys.len() / 2]
}

fn tokens(cards: &[Card]) -> Vec<&'static str> {
    cards.iter().map(|card| card.as_str()).collect()
}

#[test]
fn fixed_schedule_and_initial_mapping_match_python_contract() {
    assert_eq!(DECISION_COUNT, 10);
    assert_eq!(
        DECISION_SCHEDULE
            .iter()
            .map(|spec| (spec.street, spec.actor, spec.deal_start, spec.deal_size))
            .collect::<Vec<_>>(),
        vec![
            (Street::T0, 0, 0, 5),
            (Street::T0, 1, 5, 5),
            (Street::T1, 0, 10, 3),
            (Street::T1, 1, 13, 3),
            (Street::T2, 0, 16, 3),
            (Street::T2, 1, 19, 3),
            (Street::T3, 0, 22, 3),
            (Street::T3, 1, 25, 3),
            (Street::T4, 0, 28, 3),
            (Street::T4, 1, 31, 3),
        ]
    );

    let env = ScalarHuRlEnv::new(&ALL_CARDS).unwrap();
    let view = env.observe().unwrap();
    let observation = view.observation();
    let mapping = view.legal_action_mapping();
    assert_eq!(observation.street, Street::T0);
    assert_eq!(observation.seat, Seat::First);
    assert_eq!(observation.to_act_order, ActOrder::First);
    assert_eq!(mapping.action_count(), MAX_LEGAL_ACTIONS);
    assert_eq!(
        mapping.mask().iter().filter(|enabled| **enabled).count(),
        232
    );
    assert_eq!(
        mapping.action_set_digest(),
        "2f4c595cf062f38a89f81cf736ed48bcaab9e1884c48bbd41c61cba4d8acf41b"
    );
    assert_eq!(mapping.action_order_digest(), mapping.action_set_digest());
    assert_eq!(
        mapping.key_at(0).unwrap().to_token(),
        "rak1:0000000000000:0000000000000:000000000001f:0000000000000"
    );
    assert_eq!(
        mapping.key_at(116).unwrap().to_token(),
        PYTHON_MIDDLE_KEYS[0]
    );
    assert_eq!(
        mapping.key_at(231).unwrap().to_token(),
        "rak1:000000000001c:0000000000003:0000000000000:0000000000000"
    );
    assert_eq!(
        observation.fingerprint(),
        "218c55eee401e7034cba19ab0bbcdfb1b4124559d55babc5ec11d70303976ad6"
    );
    assert_eq!(view.digest(), PYTHON_VIEW_DIGESTS[0]);
}

#[test]
fn ten_decision_middle_policy_is_bit_exact_with_python_oracle() {
    let mut env = ScalarHuRlEnv::new(&ALL_CARDS).unwrap();
    let mut selected = Vec::new();
    for decision in 0..DECISION_COUNT {
        let view = env.observe().unwrap();
        assert_eq!(view.digest(), PYTHON_VIEW_DIGESTS[decision]);
        assert_eq!(view.public_history().len(), decision);
        let key = middle_key(&env);
        assert_eq!(key.to_token(), PYTHON_MIDDLE_KEYS[decision]);
        selected.push(key);
        let result = env.step(key).unwrap();
        assert_eq!(result.actor, decision % 2);
        assert_eq!(result.street, DECISION_SCHEDULE[decision].street);
        assert_eq!(result.done, decision + 1 == DECISION_COUNT);
        assert_eq!(
            result.rewards,
            if result.done { [0.0, -0.0] } else { [0.0, 0.0] }
        );
    }

    assert!(env.done());
    assert_eq!(env.decision_count(), 10);
    assert_eq!(env.public_history().len(), 10);
    assert!(env.boards().iter().all(|board| board.is_complete()));
    assert_eq!(tokens(&env.boards()[0].top), vec!["5h", "Qh", "5d"]);
    assert_eq!(
        tokens(&env.boards()[0].middle),
        vec!["6h", "Jd", "Kd", "4c", "6c"]
    );
    assert_eq!(
        tokens(&env.boards()[0].bottom),
        vec!["2h", "3h", "4h", "Ah", "7d"]
    );
    assert_eq!(tokens(&env.boards()[1].top), vec!["Th", "2d", "8d"]);
    assert_eq!(
        tokens(&env.boards()[1].middle),
        vec!["Jh", "Ad", "3c", "7c", "9c"]
    );
    assert_eq!(
        tokens(&env.boards()[1].bottom),
        vec!["7h", "8h", "9h", "4d", "Td"]
    );
    assert_eq!(env.terminal_rewards().unwrap(), [0.0, -0.0]);
    assert_eq!(env.terminal_rewards().unwrap().iter().sum::<f64>(), 0.0);

    let placed = env
        .boards()
        .iter()
        .flat_map(|board| board.all_cards())
        .collect::<HashSet<_>>();
    let discarded = selected
        .iter()
        .flat_map(|key| key.cards("discards").unwrap())
        .collect::<HashSet<_>>();
    assert_eq!(placed.len(), 26);
    assert_eq!(discarded.len(), 8);
    assert!(placed.is_disjoint(&discarded));
    assert_eq!(
        placed.union(&discarded).copied().collect::<HashSet<_>>(),
        ALL_CARDS[..34].iter().copied().collect::<HashSet<_>>()
    );
    assert!(env.observe().unwrap_err().message().contains("terminal"));
}

#[test]
fn pinned_rotated_deck_fixture_is_bit_exact_with_python_golden_v1() {
    let mut deck = ALL_CARDS;
    deck.rotate_left(5);
    let mut env = ScalarHuRlEnv::new(&deck).unwrap();
    for decision in 0..DECISION_COUNT {
        let view = env.observe().unwrap();
        let mapping = view.legal_action_mapping();
        assert_eq!(view.digest(), PYTHON_GOLDEN_V1_VIEW_DIGESTS[decision]);
        assert_eq!(
            mapping.action_count(),
            PYTHON_GOLDEN_V1_ACTION_COUNTS[decision]
        );
        assert_eq!(
            mapping.action_set_digest(),
            PYTHON_GOLDEN_V1_ACTION_SET_DIGESTS[decision]
        );
        assert_eq!(mapping.action_order_digest(), mapping.action_set_digest());
        let selected = mapping.key_at(mapping.action_count() / 2).unwrap();
        assert_eq!(selected.to_token(), PYTHON_GOLDEN_V1_MIDDLE_KEYS[decision]);
        env.step(selected).unwrap();
    }

    assert_eq!(env.terminal_rewards().unwrap(), [-6.0, 6.0]);
    assert_eq!(sorted_tokens(&env.boards()[0].top), vec!["Th", "4d", "Td"]);
    assert_eq!(
        sorted_tokens(&env.boards()[0].middle),
        vec!["Jh", "3c", "5c", "9c", "Jc"]
    );
    assert_eq!(
        sorted_tokens(&env.boards()[0].bottom),
        vec!["7h", "8h", "9h", "6d", "Qd"]
    );
    assert_eq!(sorted_tokens(&env.boards()[1].top), vec!["2d", "7d", "Kd"]);
    assert_eq!(
        sorted_tokens(&env.boards()[1].middle),
        vec!["3d", "6c", "8c", "Qc", "Ac"]
    );
    assert_eq!(
        sorted_tokens(&env.boards()[1].bottom),
        vec!["Qh", "Kh", "Ah", "9d", "2c"]
    );
}

#[test]
fn alternating_edge_policy_has_python_nonzero_zero_sum_reward() {
    let mut env = ScalarHuRlEnv::new(&ALL_CARDS).unwrap();
    while !env.done() {
        let mapping = env.legal_mapping().unwrap();
        let index = if env.decision_count().is_multiple_of(2) {
            0
        } else {
            mapping.action_count() - 1
        };
        env.step(mapping.key_at(index).unwrap()).unwrap();
    }
    assert_eq!(env.terminal_rewards().unwrap(), [21.0, -21.0]);
}

#[test]
fn illegal_action_and_invalid_reset_are_atomic() {
    let mut env = ScalarHuRlEnv::new(&ALL_CARDS).unwrap();
    let before = env.snapshot();
    let error = env.step(ActionKey::default()).unwrap_err();
    assert!(error.message().contains("not legal"));
    assert_eq!(env.snapshot(), before);

    let mut duplicate = ALL_CARDS;
    duplicate[51] = duplicate[0];
    let error = env.reset(&duplicate).unwrap_err();
    assert!(error.message().contains("duplicate"));
    assert_eq!(env.snapshot(), before);
    assert!(ScalarHuRlEnv::new(&ALL_CARDS[..51])
        .unwrap_err()
        .message()
        .contains("exactly 52"));
}

#[test]
fn unsupported_scoring_is_rejected_before_the_first_decision() {
    let missing_fl14 = ofc_hu_rl_engine::ScoringContext {
        fl_ev: BTreeMap::from([(13, 1.0)]),
        ..Default::default()
    };
    let error = ScalarHuRlEnv::with_scoring(&ALL_CARDS, missing_fl14).unwrap_err();
    assert!(error.message().contains("pinned Regular OFC scoring"));

    let noncanonical_float = ofc_hu_rl_engine::ScoringContext {
        fl_ev: BTreeMap::from([(14, 1e-7)]),
        ..Default::default()
    };
    let error = ScalarHuRlEnv::with_scoring(&ALL_CARDS, noncanonical_float).unwrap_err();
    assert!(error.message().contains("pinned Regular OFC scoring"));
}

#[test]
fn actor_view_never_contains_opponent_discard_or_deck_tail() {
    let mut env = ScalarHuRlEnv::new(&ALL_CARDS).unwrap();
    env.step(middle_key(&env)).unwrap();
    env.step(middle_key(&env)).unwrap();
    let first_t1 = middle_key(&env);
    let opponent_private_discard = first_t1.cards("discards").unwrap()[0];
    env.step(first_t1).unwrap();

    let second_view = env.observe().unwrap();
    let encoded = second_view.canonical_json();
    assert!(!encoded.contains(opponent_private_discard.as_str()));
    for forbidden in [
        "deck_tail",
        "remaining_deck",
        "world_state",
        "opponent_private_discard",
        "discard_mask",
    ] {
        assert!(!encoded.contains(forbidden));
    }
    for event in second_view.public_history() {
        let payload = event.to_json();
        let object = payload.as_object().unwrap();
        assert_eq!(object.len(), 7);
        assert!(!object.contains_key("discard_mask"));
    }

    env.step(middle_key(&env)).unwrap();
    let next_first = env.observe().unwrap();
    assert!(next_first
        .observation()
        .hero_private_discards
        .contains(&opponent_private_discard));
}

#[test]
fn snapshot_restore_replays_identical_views_actions_and_rewards() {
    let mut reversed = ALL_CARDS;
    reversed.reverse();
    let mut env = ScalarHuRlEnv::new(&reversed).unwrap();
    for _ in 0..4 {
        env.step(middle_key(&env)).unwrap();
    }
    let snapshot = env.snapshot();
    assert_eq!(snapshot.decision_count(), 4);
    let debug = format!("{snapshot:?}");
    assert!(debug.contains("hidden_state"));
    assert!(debug.contains("<redacted>"));
    assert!(!debug.contains("As"));

    let expected_view = env.observe().unwrap().digest();
    let mut expected_actions = Vec::new();
    while !env.done() {
        let key = middle_key(&env);
        expected_actions.push(key);
        env.step(key).unwrap();
    }
    let expected_rewards = env.terminal_rewards().unwrap();

    env.restore(&snapshot).unwrap();
    assert_eq!(env.observe().unwrap().digest(), expected_view);
    let mut replayed_actions = Vec::new();
    while !env.done() {
        let key = middle_key(&env);
        replayed_actions.push(key);
        env.step(key).unwrap();
    }
    assert_eq!(replayed_actions, expected_actions);
    assert_eq!(env.terminal_rewards().unwrap(), expected_rewards);
}

#[test]
fn actor_view_rejects_missing_history_and_wrong_row_attribution() {
    let mut env = ScalarHuRlEnv::new(&ALL_CARDS).unwrap();
    for _ in 0..4 {
        env.step(middle_key(&env)).unwrap();
    }
    let view = env.observe().unwrap();
    let observation = view.observation().clone();
    let history = view.public_history().to_vec();

    let missing = HuRlActorViewV1::from_observation(
        observation.clone(),
        history[..history.len() - 1].to_vec(),
    )
    .unwrap_err();
    assert!(missing.message().contains("complete trajectory prefix"));

    let original = &history[2];
    let mut masks = original.placement_masks();
    let source = masks.iter().position(|mask| *mask != 0).unwrap();
    let target = (source + 1) % masks.len();
    let moved_bit = masks[source] & masks[source].wrapping_neg();
    masks[source] ^= moved_bit;
    masks[target] |= moved_bit;
    let wrong_event = PublicPlacement::new(
        original.street(),
        original.acting_seat(),
        masks[0],
        masks[1],
        masks[2],
        original.discard_count(),
    )
    .unwrap();
    let mut wrong_history = history;
    wrong_history[2] = wrong_event;
    let wrong_row = HuRlActorViewV1::from_observation(observation, wrong_history).unwrap_err();
    assert!(wrong_row.message().contains("placement rows disagree"));
}

#[test]
fn action_and_view_contract_are_invariant_to_order_within_each_deal() {
    let mut permuted = ALL_CARDS;
    for spec in DECISION_SCHEDULE {
        permuted[spec.deal_start..spec.deal_start + spec.deal_size].reverse();
    }
    let mut left = ScalarHuRlEnv::new(&ALL_CARDS).unwrap();
    let mut right = ScalarHuRlEnv::new(&permuted).unwrap();
    while !left.done() {
        assert_eq!(
            left.observe().unwrap().digest(),
            right.observe().unwrap().digest()
        );
        assert_eq!(
            left.legal_mapping().unwrap(),
            right.legal_mapping().unwrap()
        );
        let selected = middle_key(&left);
        assert!(right.legal_actions().unwrap().contains(&selected));
        left.step(selected).unwrap();
        right.step(selected).unwrap();
    }
    assert_eq!(
        left.terminal_rewards().unwrap(),
        right.terminal_rewards().unwrap()
    );
    for (left_board, right_board) in left.boards().iter().zip(right.boards()) {
        assert_eq!(
            left_board.top.iter().copied().collect::<HashSet<_>>(),
            right_board.top.iter().copied().collect::<HashSet<_>>()
        );
        assert_eq!(
            left_board.middle.iter().copied().collect::<HashSet<_>>(),
            right_board.middle.iter().copied().collect::<HashSet<_>>()
        );
        assert_eq!(
            left_board.bottom.iter().copied().collect::<HashSet<_>>(),
            right_board.bottom.iter().copied().collect::<HashSet<_>>()
        );
    }
}

fn sorted_tokens(cards: &[Card]) -> Vec<&'static str> {
    let mut sorted = cards.to_vec();
    sorted.sort_unstable_by_key(|card| card.index());
    tokens(&sorted)
}
