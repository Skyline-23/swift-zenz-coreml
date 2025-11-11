import XCTest
@testable import swift_zenz_coreml

final class swift_zenz_coremlTests: XCTestCase {

    func testExample() async throws {
        // Prepare the shared environment (model + tokenizer) only once.
        let env = await makeBenchmarkEnvironment()

        // 1) Short Japanese: "ニホンゴ" → "日本語"
        await runBenchmarksFor(
            groupTag: "[ニホンゴ]",
            kanaInput: "\u{EE00}ニホンゴ\u{EE01}",
            env: env
        )

        // 2) "カンコクゴヲベンキョウスル" → "韓国語を勉強する"
        await runBenchmarksFor(
            groupTag: "[カンコクゴ]",
            kanaInput: "\u{EE00}カンコクゴヲベンキョウスル\u{EE01}",
            env: env
        )

        // 3) Long sentence: "私は今日本語の勉強をしていて〜"
        await runBenchmarksFor(
            groupTag: "[LongJP]",
            kanaInput: "\u{EE00}ワタシハイマニホンゴノベンキョウヲシテイテ、スマートフォンノキーボードデヘンカンセイドヲアゲタイトオモッテイマス\u{EE01}",
            env: env
        )

        // 4) Greeting: "オハヨウゴザイマス" → "おはようございます"
        await runBenchmarksFor(
            groupTag: "[Greet1]",
            kanaInput: "\u{EE00}オハヨウゴザイマス\u{EE01}",
            env: env
        )

        // 5) Greeting and self-introduction
        await runBenchmarksFor(
            groupTag: "[Greet2]",
            kanaInput: "\u{EE00}ハジメマシテ、ワタシハスカイラインデス\u{EE01}",
            env: env
        )

        // 6) Short question: "ゲンキデスカ"
        await runBenchmarksFor(
            groupTag: "[ShortQ]",
            kanaInput: "\u{EE00}ゲンキデスカ\u{EE01}",
            env: env
        )

        // 7) Weather: "キョウハトテモアツイデスネ"
        await runBenchmarksFor(
            groupTag: "[Weather]",
            kanaInput: "\u{EE00}キョウハトテモアツイデスネ\u{EE01}",
            env: env
        )

        // 8) Meeting: "アシタノゴゴサンジニエキデアイマショウ"
        await runBenchmarksFor(
            groupTag: "[Meetup]",
            kanaInput: "\u{EE00}アシタノゴゴサンジニエキデアイマショウ\u{EE01}",
            env: env
        )

        // 9) Dinner: "キョウノバンナニヲタベタイデスカ"
        await runBenchmarksFor(
            groupTag: "[Dinner]",
            kanaInput: "\u{EE00}キョウノバンナニヲタベタイデスカ\u{EE01}",
            env: env
        )

        // 10) Japanese culture: "ニホンノブンカニキョウミガアリマス"
        await runBenchmarksFor(
            groupTag: "[Culture]",
            kanaInput: "\u{EE00}ニホンノブンカニキョウミガアリマス\u{EE01}",
            env: env
        )

        // 11) Korean language skill: "カンコクゴヲモットジョウズニハナセルヨウニナリタイデス"
        await runBenchmarksFor(
            groupTag: "[KoreanSkill]",
            kanaInput: "\u{EE00}カンコクゴヲモットジョウズニハナセルヨウニナリタイデス\u{EE01}",
            env: env
        )

        // 12) Hobby: "ヒマナトキハヨクエイガヲミマス"
        await runBenchmarksFor(
            groupTag: "[HobbyMovie]",
            kanaInput: "\u{EE00}ヒマナトキハヨクエイガヲミマス\u{EE01}",
            env: env
        )

        // 13) Hobby (Reading): "ワタシノシュミハホンヲヨムコトデス"
        await runBenchmarksFor(
            groupTag: "[HobbyBook]",
            kanaInput: "\u{EE00}ワタシノシュミハホンヲヨムコトデス\u{EE01}",
            env: env
        )

        // 14) Computer issue: "コンピュータノガメンガフリーズシテシマイマシタ"
        await runBenchmarksFor(
            groupTag: "[PCFreeze]",
            kanaInput: "\u{EE00}コンピュータノガメンガフリーズシテシマイマシタ\u{EE01}",
            env: env
        )

        // 15) Battery: "スマホノバッテリーガスグニナクナッテコマッテイマス"
        await runBenchmarksFor(
            groupTag: "[Battery]",
            kanaInput: "\u{EE00}スマホノバッテリーガスグニナクナッテコマッテイマス\u{EE01}",
            env: env
        )

        // 16) Keyboard accuracy: "キーボードノヘンカンセイドガアガルト…"
        await runBenchmarksFor(
            groupTag: "[Keyboard]",
            kanaInput: "\u{EE00}キーボードノヘンカンセイドガアガルトモットハヤクウテマス\u{EE01}",
            env: env
        )

        // 17) Cafe meeting: "キノウハトモダチトエキマエノカフェデ…"
        await runBenchmarksFor(
            groupTag: "[Cafe]",
            kanaInput: "\u{EE00}キノウハトモダチトエキマエノカフェデコーヒーヲノミマシタ\u{EE01}",
            env: env
        )

        // 18) Schedule: "サンジニシゴトガオワルノデヨジニアエマス"
        await runBenchmarksFor(
            groupTag: "[TimeMeet]",
            kanaInput: "\u{EE00}サンジニシゴトガオワルノデヨジニアエマス\u{EE01}",
            env: env
        )

        // 19) Next holiday: "ツギノヤスミハドコニイキマショウカ"
        await runBenchmarksFor(
            groupTag: "[NextHoliday]",
            kanaInput: "\u{EE00}ツギノヤスミハドコニイキマショウカ\u{EE01}",
            env: env
        )

        // 20) LongJP-2: Hobby + long sentence
        await runBenchmarksFor(
            groupTag: "[LongJP2]",
            kanaInput: "\u{EE00}ワタシノシュミハホンヲヨムコトデ、トクニミステリーショウセツガスキデス\u{EE01}",
            env: env
        )

        // 21) LongJP-3: Daily routine
        await runBenchmarksFor(
            groupTag: "[LongJP3]",
            kanaInput: "\u{EE00}マイニチシゴトノマエニコーヒーヲイッパイノムノガナンタノシミデス\u{EE01}",
            env: env
        )

        // 22) LongJP-4: Study + time reference
        await runBenchmarksFor(
            groupTag: "[LongJP4]",
            kanaInput: "\u{EE00}ワタシハマイニチネルトキニニジカンホドニホンゴノベンキョウヲシテイマス\u{EE01}",
            env: env
        )

        // 23) LongJP-Keyboard: Keyboard context enhancement
        await runBenchmarksFor(
            groupTag: "[LongJPKeyboard]",
            kanaInput: "\u{EE00}イツモスマートフォンノキーボードデニホンゴヲウツノデ、ヘンカンセイドガタカイトホントウニタスカリマス\u{EE01}",
            env: env
        )
    }
}
