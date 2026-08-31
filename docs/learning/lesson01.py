# -*- coding: utf-8 -*-
"""
レッスン01: 期待値とモンテカルロ
--------------------------------
実行方法:  python lesson01.py

標準ライブラリだけで動きます（インストール不要）。
所要 10〜20 秒程度。
"""
import math
import random
from statistics import mean, stdev

random.seed(0)  # 再現性のため。演習では変えてみること


def avg_abs_error(sampler, n, true_value, repeats):
    """同じ n で repeats 回試して、誤差の平均を返す（1回だけだと運で決まるため）。"""
    return mean(abs(sampler(n) - true_value) for _ in range(repeats))


# ══════════════════════════════════════════════════════════════
# Part 1: 期待値の定義（サイコロ）
# ══════════════════════════════════════════════════════════════
def dice_mc(n):
    return mean(random.randint(1, 6) for _ in range(n))


def part1():
    print("=" * 64)
    print("Part 1: サイコロの期待値 — 大数の法則")
    print("=" * 64)

    # 手計算: E[X] = Σ p(x)·x = (1+2+3+4+5+6)/6
    exact = sum(x * (1 / 6) for x in range(1, 7))
    print(f"厳密な期待値  E[X] = Σ p(x)·x = {exact:.4f}\n")

    print(f"{'n':>8} {'誤差の平均(50回試行)':>22}")
    print("-" * 32)
    for n in (10, 100, 1_000, 10_000):
        err = avg_abs_error(dice_mc, n, exact, repeats=50)
        print(f"{n:>8} {err:>22.5f}")
    print()
    print("→ n を 100 倍にすると誤差はおよそ 1/10 になる。")
    print("  1回だけ試すと運で決まるので、必ず複数回の平均で見ること。")
    print()


# ══════════════════════════════════════════════════════════════
# Part 2: OFC のトイ問題 — この選択の価値はいくらか
# ══════════════════════════════════════════════════════════════
#
# 状況（簡略化した OFC Pineapple の 1 局面）:
#
#   自分の bottom 行に ♥ が 4 枚。残り 1 スロット。
#   次のターンに 3 枚配られる。その中に ♥ かジョーカーが 1 枚でもあれば
#   フラッシュが完成し、bottom のロイヤリティ +4 点。
#   （+4 という値は ai/engine/game_engine.py の get_bottom_royalty より）
#
#   未確認カード 44 枚。うち ♥ が 9 枚、ジョーカー(X1/X2) が 2 枚。
#   → フラッシュを完成させるカード（アウツ）は 11 枚。
#
UNSEEN = 44        # 未確認カードの枚数
OUTS = 11          # 完成札（♥9 + ジョーカー2）
DRAW = 3           # 次のターンに配られる枚数
FLUSH_ROYALTY = 4  # bottom のフラッシュ = +4 点


def exact_ev():
    """厳密計算。「1枚も来ない確率」を組合せで出し、その余事象を取る。"""
    p_miss = math.comb(UNSEEN - OUTS, DRAW) / math.comb(UNSEEN, DRAW)
    p_hit = 1.0 - p_miss
    return p_hit * FLUSH_ROYALTY, p_hit


def mc_ev(n):
    """モンテカルロ。実際に 3 枚引く、を n 回繰り返して平均する。
    カードに 0..43 の番号を振り、0..OUTS-1 を「完成札」とみなす。"""
    hits = 0
    for _ in range(n):
        if min(random.sample(range(UNSEEN), DRAW)) < OUTS:
            hits += 1
    return FLUSH_ROYALTY * hits / n


def part2():
    print("=" * 64)
    print("Part 2: OFC トイ問題 — フラッシュ・ドローの期待値")
    print("=" * 64)
    ev, p = exact_ev()
    print(f"未確認 {UNSEEN} 枚 / アウツ {OUTS} 枚 / {DRAW} 枚引く")
    print(f"  完成確率 = 1 - C({UNSEEN-OUTS},{DRAW}) / C({UNSEEN},{DRAW})")
    print(f"           = 1 - {math.comb(UNSEEN-OUTS,DRAW)} / {math.comb(UNSEEN,DRAW)}")
    print(f"           = {p:.6f}")
    print(f"  期待値   = {p:.6f} × {FLUSH_ROYALTY} = {ev:.4f} 点\n")
    print(f"  モンテカルロ (n=200,000) = {mc_ev(200_000):.4f} 点  ← 厳密値と一致するはず\n")
    print("比較: ストレート・ドロー（完成すれば +2点、完成確率 0.80）なら")
    print(f"      期待値 = 0.80 × 2 = {0.80*2:.4f} 点")
    print("→ この局面はフラッシュ狙いが正解。これが「EV で手を選ぶ」ということ。")
    print("  あなたのプロジェクトの T0〜T4 の全ターンが、原理的にはこれと同じ計算です。")
    print()


# ══════════════════════════════════════════════════════════════
# Part 3: 誤差は 1/√n で縮む
# ══════════════════════════════════════════════════════════════
def part3():
    print("=" * 64)
    print("Part 3: サンプル数と誤差 — 1/√n の法則")
    print("=" * 64)
    ev_true, p = exact_ev()

    # 理論値: 1回の観測は 0 か 4 のどちらか。σ = 4·√(p(1-p))
    sigma = FLUSH_ROYALTY * math.sqrt(p * (1 - p))
    print(f"1 サンプルの標準偏差 σ = {FLUSH_ROYALTY}·√(p(1-p)) = {sigma:.4f}")
    print(f"理論上の標準誤差 SE = σ / √n\n")

    print(f"{'n':>8} {'実測SE':>10} {'理論SE=σ/√n':>14} {'比':>8}")
    print("-" * 44)
    for n, reps in ((10, 400), (100, 400), (1_000, 200), (10_000, 60)):
        ests = [mc_ev(n) for _ in range(reps)]
        emp = stdev(ests)
        theo = sigma / math.sqrt(n)
        print(f"{n:>8} {emp:>10.5f} {theo:>14.5f} {emp/theo:>8.3f}")
    print()
    print("→ 実測と理論がほぼ一致（比が 1.0 付近）。誤差は 1/√n で縮む。")
    print("→ 誤差を 1/10 にしたければ サンプルを 100 倍。ここが GCP 代の正体です。")
    print()


# ══════════════════════════════════════════════════════════════
# Part 4: バイアスはサンプル数では消えない（このプロジェクト最重要の教訓）
# ══════════════════════════════════════════════════════════════
def mc_ev_biased(n, bias=0.3):
    """評価器が常に bias 点だけ高く見積もってしまう場合。"""
    return mc_ev(n) + bias


def part4():
    print("=" * 64)
    print("Part 4: バイアスはサンプル数では消えない")
    print("=" * 64)
    ev_true, _ = exact_ev()
    print(f"真の期待値 = {ev_true:.4f}\n")
    print(f"{'n':>8} | {'正しい評価器の誤差':>20} | {'偏った評価器の誤差':>20}")
    print("-" * 56)
    for n, reps in ((100, 200), (1_000, 100), (10_000, 40), (100_000, 10)):
        good = avg_abs_error(mc_ev, n, ev_true, repeats=reps)
        bad = avg_abs_error(mc_ev_biased, n, ev_true, repeats=reps)
        print(f"{n:>8} | {good:>20.5f} | {bad:>20.5f}")
    print()
    print("→ 左は n とともに 0 へ向かう（分散はサンプルで潰せる）")
    print("→ 右は 0.3 で止まる（バイアスはサンプルでは潰せない）")
    print()
    print("これが value_chain_milestones_20260731.md の")
    print('  「評価の誤りは探索量で救えない（20,000シミュレーション実験）」')
    print("の正体です。")
    print()
    print("★ここが今日の一番大事な結論★")
    print("  『精度が足りないのでサンプル数を増やしましょう』という提案が正しいのは、")
    print("  誤差が分散由来のときだけ。バイアス由来なら、いくら回しても無駄です。")
    print("  LLM の提案を評価するとき、まずこの切り分けを聞いてください。")
    print()


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    print("=" * 64)
    print("演習は 「レッスン01_期待値とモンテカルロ.md」 を参照してください。")
    print("=" * 64)
