# R005E2A geometry-matched donor preflight bundle

状态：`PASS_DONOR_MANIFEST_ONLY__NO_SCIENTIFIC_TRAINING`。

- server run：`/data/khsong/aorta_runs/r005e2a_998d85a`；
- frozen source commit：`998d85a`；
- source archive SHA-256：
  `a85f2eab0916b78a5992537320d1f51478bad05a37b9431ee81490d504a8d6a8`；
- source-controlled protocol tests：28/28 PASS；
- panel resolution：671个base artifacts + 1个R005E1A replacement artifact，均由预登记SHA精确解析；
- donor manifest：672 anchors、6 folds、24 stages、7,392 assignments；
- constraints：全部recipient逐stage精确覆盖、donor只来自注册训练池、跨recording group、stage内
  无放回；`selection_fit`与`refit_train`的donor visual marginal均为精确置换；
- primary/replay：byte-exact，二者SHA-256均为
  `9a5f4878e9856a6cb020c1d814f034233aa7b6585cc2507af16ab29cb2373093`；
- independent mechanical audit：PASS，SHA-256为
  `5c05ccbd320b00e0e4388300b4f23e0411f384a58e635c664abc8c902f0d519d`；
- complete compressed server output bundle：
  `r005e2a_outputs.tar.gz`，SHA-256
  `be33cb9710a90363bdd209e09ea9c582abf7afbe2308296d9f249109a129ae3c`。

可读的primary manifest保存为
`experiments/R005/inputs/r005e2_mismatch_donor_manifest.json`；压缩bundle内同时保留primary、replay与
audit。该preflight只证明V1错配控制的构造合法且可重放，没有读取或产生correct/mismatch/removed
模型性能，不提供视觉可恢复性、架构创新、外观鲁棒性或驾驶收益证据。
