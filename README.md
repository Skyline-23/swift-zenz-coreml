# Swift-zenz-CoreML

🇯🇵 Swiftで[zenz-CoreML](https://github.com/Skyline-23/zenz-CoreML)を使用するサンプルリポジトリです。  
🇰🇷 Swift에서 [zenz-CoreML](https://github.com/Skyline-23/zenz-CoreML)을 사용하는 샘플 리포지토리입니다.  
🇺🇸 Sample repository demonstrating how to use [zenz-CoreML](https://github.com/Skyline-23/zenz-CoreML) with Swift.  

Sample repository using [Skyline-23/zenz-CoreML](https://github.com/Skyline-23/zenz-CoreML) with Swift.

### クローン / 클론 / Clone

```bash
git clone https://github.com/ensan-hcl/swift-zenz-coreml --recursive
```

## 実行 / 실행 / Run

```bash
xcodebuild -scheme swift-zenz-coreml -destination "platform=macOS,name=Any Mac" test
```

## ベンチマーク (Core ML greedy decoding) / 벤치마크 (Core ML greedy decoding) / Benchmarks (Core ML greedy decoding)

## 1回目＋2回目の統合平均 / 1회차 + 2회차 통합 평균 / Combined Average (Round 1 + 2)

Tested on MacBook Pro (14-inch, 2023) with Apple M3 Pro chip.

| Strategy | Avg latency (s) |
|----------|----------------:|
| Stateful / Async global | 0.169 |
| Stateful / Sync main | 0.167 |
| Stateless / Async global | 0.169 |
| Stateless / Sync main | 0.163 |

## 文の長さ別平均 / 문장 길이별 통합 평균 / Average by Sentence Length

**短い入力 (≤ 25 tokens) / 짧은 입력 (≤ 25 tokens) / Short Inputs (≤ 25 tokens)**  
| Strategy | Avg latency (s) |
|----------|-----------------:|
| Stateful / Async global | 0.094 |
| Stateful / Sync main | 0.091 |
| Stateless / Async global | 0.096 |
| Stateless / Sync main | 0.091 |

**長い入力 (> 25 tokens) / 긴 입력 (> 25 tokens) / Long Inputs (> 25 tokens)**  
| Strategy | Avg latency (s) |
|----------|-----------------:|
| Stateful / Async global | 0.246 |
| Stateful / Sync main | 0.241 |
| Stateless / Async global | 0.244 |
| Stateless / Sync main | 0.240 |

Detailed benchmark results for Round 1 and Round 2 are available here:  
[Round 1 details](benchmarks/round1.md)  
[Round 2 details](benchmarks/round2.md)
