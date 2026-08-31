# M3 behavior sharded-collection E2E smoke (2026-07-13)

This is a deliberately small, non-promoting end-to-end check of the
append-only sharded collector with the four exact-hash HU T1/T2 PolicyValueNet
routes. It is not the production calibration population and is not strategic
strength evidence.

## Natural population

- roots: 2
- decisions: 8
- shards: 2 (one root each)
- collection content SHA-256:
  `9d60a5296ebc1a9c9b0d2fa877d541267e57f680e79e07d7fa63386488cc0b93`
- layout SHA-256:
  `71a06ecf57c33999b3d04de287c9a6bf90b25cc7d7433761537ba894033e448f`
- top manifest SHA-256:
  `13c30e49301b925abba3d9f908296f512d3b3f54d4529d58d09e71ad306877bf`

## Targeted Joker population

- roots: 12
- decisions: 48
- shards: 3 (four roots each)
- all 12 `T1/T2 x BB/BTN x Joker 0/1/2` target cells: one root each
- collection content SHA-256:
  `5fbeb0e8c29512c1910462f117b378d5e007fb5a10c76d38fb3995d02557a3f5`
- layout SHA-256:
  `a11ac7888f5c483f58e6946a4f04f464e2c533f24663fae5edb8111773455928`
- top manifest SHA-256:
  `2679e263cf35b377d252e64105ce28472ab4b39fed10ccb45f8d74f82343f83a`

Every published shard and the top manifest passed fresh readback. Raw
collection artifacts keep `promotion_eligible=false`; the production plan is
stored separately in `ai/reports/m3_behavior_collection_plan_20260713`.
