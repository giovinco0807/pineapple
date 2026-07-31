import json, sys, time
from pathlib import Path
import numpy as np
import ai.tutor.exact_late as exact_late
import ai.tutor.t3_second_features as F
from ai.engine.action_space import get_turn_actions
from ai.tutor.t3_second_solver import _board

src = Path('D:/ofc_data/t3_second_teacher_50k')
out = Path('D:/ofc_data/t3_second_train_50k'); out.mkdir(parents=True, exist_ok=True)
manifest = {}
for name in ('fit','dev','test'):
    xs, ys, js = [], [], []
    t0 = time.time()
    with (src/f'{name}.jsonl').open(encoding='utf-8') as fh:
        for line_no, line in enumerate(fh):
            if not line.strip(): continue
            r = json.loads(line)
            cache = F.T3NodeCache(r['bb_board'], r['btn_board'], r['btn_dead'], r['draw'])
            btn = _board(r['btn_board'])
            by_key = {a['action_key']: a['value'] for a in r['actions']}
            jok = sum(1 for rows in (r['bb_board'], r['btn_board']) for row in rows for c in row if c in ('X1','X2')) \
                  + sum(1 for c in r['draw'] if c in ('X1','X2'))
            for a in get_turn_actions(list(r['draw']), btn):
                k = exact_late.action_key(a)
                if k not in by_key: continue
                af = exact_late.apply_action(btn, a)
                xs.append(F.encode_action((af.top, af.middle, af.bottom), cache))
                ys.append(by_key[k]); js.append(jok)
            if (line_no+1) % 5000 == 0:
                print(f'  {name} {line_no+1} rows={len(xs)} {time.time()-t0:.0f}s', flush=True)
    x = np.asarray(xs, dtype=np.float32); y = np.asarray(ys, dtype=np.float32); j = np.asarray(js, dtype=np.int8)
    np.savez_compressed(out/f'{name}.npz', x=x, y=y, jokers=j)
    manifest[name] = {'rows': int(x.shape[0]), 'mean': float(y.mean()), 'std': float(y.std())}
    print(name, manifest[name], flush=True)
(out/'manifest.json').write_text(json.dumps({'schema':'ofc_t3_second_train/v1','feature_size':F.FEATURE_SIZE,
    'source':str(src),'target_is_exact':False,'splits':manifest}, indent=2), encoding='utf-8')
print('done')
