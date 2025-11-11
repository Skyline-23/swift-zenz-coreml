// The Swift Programming Language
// https://docs.swift.org/swift-book

import CoreML
import Tokenizers
import Foundation

struct BenchmarkResult {
    let label: String
    let duration: TimeInterval
}

// CoreML 모델 로드 함수
// Load the CoreML model
func loadModel() -> zenz_v1? {
    let config = MLModelConfiguration()
    return try? zenz_v1(configuration: config)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)
func loadStatefulModel() -> zenz_v1_stateful? {
    do {
        let config = MLModelConfiguration()
        // KR: 상태를 가지는 Core ML 모델을 로드하고, CPU+GPU 둘 다 활용하도록 설정합니다.
        // JP: ステートフルな Core ML モデルを読み込み、CPU と GPU の両方を使うように設定します。
        // EN: Load the stateful Core ML model and configure it to use both CPU and GPU.
        config.computeUnits = .cpuAndGPU
        return try zenz_v1_stateful(configuration: config)
    } catch let error {
        print(error)
        return nil
    }
}

// Load the Tokenizer model
func loadTokenizer() async -> Tokenizer? {
    guard let modelFolder = Bundle.module.resourceURL else {
        print("Model Folder was not found")
        return nil
    }
    do {
        return try await AutoTokenizer.from(modelFolder: modelFolder)
    } catch {
        fatalError(error.localizedDescription)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)
func predictStateful(text: String, model: zenz_v1_stateful, tokenizer: Tokenizer) -> [String] {
    let state = model.makeState()
    
    // 텍스트를 토크나이저를 사용하여 인코딩
    // Encode the input text using the tokenizer
    let inputIDs = tokenizer.encode(text: text)
    print("[Stateful Predict] inputIDs:", text, inputIDs)
    
    // 입력을 위한 MLMultiArray 생성 (Int32)
    // Create MLMultiArray for input (Int32)
    let inputArray = try? MLMultiArray(shape: [1, 16], dataType: .int32)
    for (index, token) in inputIDs.enumerated() {
        inputArray?[index] = NSNumber(value: token)
    }
    
    // Attention mask 생성 (Int32, 1/0)
    // Create attention mask (Int32, 1/0)
    let attentionMask = try? MLMultiArray(shape: [1, 16], dataType: .int32)
    for i in 0..<inputIDs.count {
        attentionMask?[i] = 1
    }
    
    guard let inputArray, let attentionMask else { return [] }
    
    // Core ML stateful 입력 타입 사용
    // Use Core ML stateful input type
    let input = zenz_v1_statefulInput(input_ids: inputArray, attention_mask: attentionMask)
    
    // stateful prediction
    let output = try? model.prediction(input: input, using: state)
    
    // 출력 logits 디코딩 (output → logits)
    // Decode logits (output → logits)
    let logits = output?.logits
    
    guard let logits else { return [] }
    
    var predictedTokenIDs = [[Int]]()
    for batchID in 0..<logits.shape[0].intValue {
        predictedTokenIDs.append([])
        for i in 0..<logits.shape[1].intValue {
            let maxId = (0..<6000).max {
                logits[[batchID, i, $0] as [NSNumber]].floatValue <
                logits[[0, i, $1] as [NSNumber]].floatValue
            } ?? 0
            predictedTokenIDs[batchID].append(maxId)
        }
    }
    
    print(predictedTokenIDs)
    let predictedTexts = predictedTokenIDs.map { tokenizer.decode(tokens: $0) }
    return predictedTexts
}

// 예측 수행 함수
// Perform prediction
func predict(text: String, model: zenz_v1, tokenizer: Tokenizer) -> [String] {
    // 텍스트를 토크나이저를 사용하여 인코딩
    // Encode the input text using the tokenizer
    let inputIDs = tokenizer.encode(text: text)
    print("[Stateless Predict][Sync] inputIDs:", text, inputIDs)
    
    // 입력을 위한 MLMultiArray 생성
    // Create MLMultiArray for input
    let inputArray = try? MLMultiArray(shape: [1, 16], dataType: .float32)
    for (index, token) in inputIDs.enumerated() {
        inputArray?[index] = NSNumber(value: token)
    }
    
    guard let inputArray else { return [] }
    // 모델 입력 생성 (attention_mask 없이 input_ids만 전달)
    // Create model input (only input_ids, no attention_mask)
    let input = zenz_v1Input(input_ids: inputArray)
    
    // 예측 수행
    // Perform prediction
    let output = try? model.prediction(input: input)
    
    // 출력 logits 디코딩
    // Decode the output logits
    let logits = output?.logits
    
    guard let logits else { return [] }
    
    // logits에서 예측된 토큰 ID 추출
    // Extract predicted token IDs from logits
    var predictedTokenIDs = [[Int]]()
    for batchID in 0..<logits.shape[0].intValue {
        predictedTokenIDs.append([])
        for i in 0..<logits.shape[1].intValue {
            var logitValues = [Float]()
            // get argMax
            let maxId = (0..<6000).max {
                logits[[batchID, i, $0] as [NSNumber]].floatValue < logits[[0, i, $1] as [NSNumber]].floatValue
            } ?? 0
            predictedTokenIDs[batchID].append(maxId)
        }
    }
    
    // 예측된 토큰 ID를 다시 텍스트로 디코딩
    // Decode the predicted token IDs back to text
    print(predictedTokenIDs)
    let predictedTexts = predictedTokenIDs.map { tokenizer.decode(tokens: $0) }
    
    // 결과 출력
    // Print the result
    return predictedTexts
}

// 예측 수행 함수
// Perform prediction
@available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, *)
func predict(text: String, model: zenz_v1, tokenizer: Tokenizer) async -> [String] {
    // 텍스트를 토크나이저를 사용하여 인코딩
    // Encode the input text using the tokenizer
    let inputIDs = tokenizer.encode(text: text)
    print("[Stateless Predict][Async] inputIDs:", text, inputIDs)
    
    // 입력을 위한 MLMultiArray 생성
    // Create MLMultiArray for input
    let inputArray = try? MLMultiArray(shape: [1, 16], dataType: .float32)
    for (index, token) in inputIDs.enumerated() {
        inputArray?[index] = NSNumber(value: token)
    }
    
    guard let inputArray else { return [] }
    // 모델 입력 생성 (attention_mask 없이 input_ids만 전달)
    // Create model input (only input_ids, no attention_mask)
    let input = zenz_v1Input(input_ids: inputArray)
    
    // 예측 수행
    // Perform prediction
    let output = try? await model.prediction(input: input)
    
    // 출력 logits 디코딩
    // Decode the output logits
    let logits = output?.logits
    
    guard let logits else { return [] }
    
    // logits에서 예측된 토큰 ID 추출
    // Extract predicted token IDs from logits
    var predictedTokenIDs = [[Int]]()
    for batchID in 0..<logits.shape[0].intValue {
        predictedTokenIDs.append([])
        for i in 0..<logits.shape[1].intValue {
            var logitValues = [Float]()
            // get argMax
            let maxId = (0..<6000).max {
                logits[[batchID, i, $0] as [NSNumber]].floatValue < logits[[0, i, $1] as [NSNumber]].floatValue
            } ?? 0
            predictedTokenIDs[batchID].append(maxId)
        }
    }
    
    // 예측된 토큰 ID를 다시 텍스트로 디코딩
    // Decode the predicted token IDs back to text
    print(predictedTokenIDs)
    let predictedTexts = predictedTokenIDs.map { tokenizer.decode(tokens: $0) }
    
    // 결과 출력
    // Print the result
    return predictedTexts
}

@available(macOS, deprecated: 10.14, message: "Use newer API predict(text:model:tokenizer) async")
@available(iOS, deprecated: 16.0, message: "Use newer API predict(text:model:tokenizer) async")
@available(tvOS, deprecated: 16.0, message: "Use newer API predict(text:model:tokenizer) async")
@available(watchOS, deprecated: 9.0, message: "Use newer API predict(text:model:tokenizer) async")
func predictDispatch(text: String, model: zenz_v1, tokenizer: Tokenizer, qos: DispatchQoS) async -> [String] {
    return await withCheckedContinuation { continuation in
        DispatchQueue.global(qos: qos.qosClass).async {
            let result = predict(text: text, model: model, tokenizer: tokenizer)
            continuation.resume(returning: result)
        }
    }
}

// KR: Stateful Core ML 모델과 KV 캐시를 사용해서 Greedy Search로 토큰을 한 단계씩 생성합니다.
// JP: ステートフルな Core ML モデルと KV キャッシュを使い、Greedy サーチでトークンを一つずつ生成します。
// EN: Perform greedy token-by-token generation using the stateful Core ML model and its KV cache.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)
func greedyPredictStateful(text: String, model: zenz_v1_stateful, tokenizer: Tokenizer) -> String {
    let state = model.makeState()
    // KR: Core ML이 관리하는 state 객체로, keyCache / valueCache / pastLen을 포함합니다.
    // JP: Core ML が管理する state オブジェクトで、keyCache / valueCache / pastLen を内部に持ちます。
    // EN: Core ML-managed state object that internally holds keyCache / valueCache / pastLen.
    
    var generatedIDs = tokenizer.encode(text: text)
    print("[Stateful Greedy] inputIDs:", text, generatedIDs)
    
    let batchSize = 1
    let maxSeqLength = 128
    let eosTokenID: Int32 = 3  // 모델에 맞게 조정 필요
    
    // KR: (1) 프롬프트 전체를 한 번 넣어서 KV 캐시를 초기화합니다.
    // JP: (1) 入力プロンプト全体を一度通して、KV キャッシュを初期化します。
    // EN: (1) Feed the whole prompt once to initialize the KV cache.
    do {
        let seqLen = generatedIDs.count
        guard let inputArray = try? MLMultiArray(
                  shape: [NSNumber(value: batchSize), NSNumber(value: seqLen)],
                  dataType: .int32
              ),
              let attentionMask = try? MLMultiArray(
                  shape: [NSNumber(value: batchSize), NSNumber(value: seqLen)],
                  dataType: .int32
              ) else {
            return ""
        }
        
        for (index, token) in generatedIDs.enumerated() {
            inputArray[index] = NSNumber(value: token)
            attentionMask[index] = 1
        }
        
        let input = zenz_v1_statefulInput(input_ids: inputArray, attention_mask: attentionMask)
        guard let output = try? model.prediction(input: input, using: state) else {
            return ""
        }
        
        let logits = output.logits
        let lastIndex = seqLen - 1
        let vocabSize = logits.shape[2].intValue
        
        var bestID: Int32 = 0
        var bestScore: Float = -.infinity
        
        for v in 0..<vocabSize {
            let score = logits[[0, lastIndex, v] as [NSNumber]].floatValue
            if score > bestScore {
                bestScore = score
                bestID = Int32(v)
            }
        }
        
        generatedIDs.append(Int(bestID))
        if bestID == eosTokenID {
            return tokenizer.decode(tokens: generatedIDs)
        }
    }
    
    // (2) 증분 디코딩용 [1,1] 버퍼를 한 번만 생성하고 재사용
    guard let stepInputArray = try? MLMultiArray(
              shape: [NSNumber(value: batchSize), 1],
              dataType: .int32
          ),
          let stepAttentionMask = try? MLMultiArray(
              shape: [NSNumber(value: batchSize), 1],
              dataType: .int32
          ) else {
        return ""
    }
    
    // KR: 이후에는 마지막 토큰만 [1,1] 입력으로 넣어가며 한 토큰씩 greedy로 확장합니다.
    // JP: 以降は最後のトークンだけを [1,1] 入力として与え、1 トークンずつ Greedy に生成を延長します。
    // EN: After initialization, we only feed the last token as a [1,1] input and extend the sequence greedily one token at a time.
    while generatedIDs.count < maxSeqLength {
        guard let lastTokenInt = generatedIDs.last else {
            break
        }
        
        let lastTokenID = Int32(lastTokenInt)
        if lastTokenID == eosTokenID {
            break
        }
        
        // 여기서는 할당 없이 값만 덮어쓰기
        stepInputArray[0] = NSNumber(value: lastTokenID)
        stepAttentionMask[0] = 1
        
        let input = zenz_v1_statefulInput(input_ids: stepInputArray, attention_mask: stepAttentionMask)
        guard let output = try? model.prediction(input: input, using: state) else {
            break
        }
        
        let logits = output.logits
        let vocabSize = logits.shape[2].intValue
        
        var bestID: Int32 = 0
        var bestScore: Float = -.infinity
        
        for v in 0..<vocabSize {
            let score = logits[[0, 0, v] as [NSNumber]].floatValue
            if score > bestScore {
                bestScore = score
                bestID = Int32(v)
            }
        }
        
        generatedIDs.append(Int(bestID))
        if bestID == eosTokenID {
            break
        }
    }
    
    let predictedText = tokenizer.decode(tokens: generatedIDs)
    let cleanedText = predictedText.replacingOccurrences(of: "[PAD]", with: "")
    return cleanedText
}

// Greedy search를 사용하여 예측 수행
// Perform prediction using Greedy search
func greedyPredict(text: String, model: zenz_v1, tokenizer: Tokenizer) -> String {
    // 텍스트를 토크나이저를 사용하여 인코딩
    // Encode the input text using the tokenizer
    var inputIDs = tokenizer.encode(text: text)
    print("[Stateless Greedy][Sync] inputIDs:", text, inputIDs)
    
    // 최대 시퀀스 길이 설정
    // Set the maximum sequence length
    let maxSeqLength = 128
    let batchSize = 1
    
    // 예측된 토큰 ID를 저장할 배열
    // Array to store predicted token IDs
    var predictedTokenIDs = inputIDs
    
    while true {
        // 입력을 위한 MLMultiArray 생성
        // Create MLMultiArray for input
        let inputArray = try? MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: predictedTokenIDs.count)], dataType: .int32)
        for (index, token) in predictedTokenIDs.enumerated() {
            inputArray?[index] = NSNumber(value: token)
        }
        
        guard let inputArray else { return "" }
        
        // 모델 입력 생성 (attention_mask 없이 input_ids만 전달)
        // Create model input (only input_ids, no attention_mask)
        let input = zenz_v1Input(input_ids: inputArray)
        
        // 예측 수행
        // Perform prediction
        guard let output = try? model.prediction(input: input) else { return "" }
        
        // 출력 logits 디코딩
        // Decode the output logits
        let logits = output.logits
        
        // logits에서 예측된 토큰 ID 추출
        // Extract predicted token ID from logits
        let nextTokenID = (0..<logits.shape[2].intValue).max {
            logits[[0, predictedTokenIDs.count - 1, $0] as [NSNumber]].floatValue <
                logits[[0, predictedTokenIDs.count - 1, $1] as [NSNumber]].floatValue
        } ?? 0
        
        // 종료 토큰 체크 (예: <EOS> 토큰 ID)
        // Check for end token (e.g., <EOS> token ID)
        if nextTokenID == 3 {
            break
        }
        
        // 예측된 토큰 ID를 추가
        // Add the predicted token ID
        predictedTokenIDs.append(nextTokenID)
        
        // 최대 시퀀스 길이에 도달하면 종료
        // Exit if the maximum sequence length is reached
        if predictedTokenIDs.count >= maxSeqLength {
            break
        }
    }
    
    // 예측된 토큰 ID를 다시 텍스트로 디코딩
    // Decode the predicted token IDs back to text
    let predictedText = tokenizer.decode(tokens: predictedTokenIDs)
    
    // 결과 출력
    // Print the result
    return predictedText
}

// Greedy search를 사용하여 예측 수행
// Perform prediction using Greedy search
@available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, *)
func greedyPredict(text: String, model: zenz_v1, tokenizer: Tokenizer) async -> String {
    // 텍스트를 토크나이저를 사용하여 인코딩
    // Encode the input text using the tokenizer
    var inputIDs = tokenizer.encode(text: text)
    print("[Stateless Greedy][Async] inputIDs:", text, inputIDs)
    
    // 최대 시퀀스 길이 설정
    // Set the maximum sequence length
    let maxSeqLength = 128
    let batchSize = 1
    
    // 예측된 토큰 ID를 저장할 배열
    // Array to store predicted token IDs
    var predictedTokenIDs = inputIDs
    
    while true {
        // 입력을 위한 MLMultiArray 생성
        // Create MLMultiArray for input
        let inputArray = try? MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: predictedTokenIDs.count)], dataType: .int32)
        for (index, token) in predictedTokenIDs.enumerated() {
            inputArray?[index] = NSNumber(value: token)
        }
        
        guard let inputArray else { return "" }
        
        // 모델 입력 생성 (attention_mask 없이 input_ids만 전달)
        // Create model input (only input_ids, no attention_mask)
        let input = zenz_v1Input(input_ids: inputArray)
        
        // 예측 수행
        // Perform prediction
        guard let output = try? await model.prediction(input: input) else { return "" }
        
        // 출력 logits 디코딩
        // Decode the output logits
        let logits = output.logits
        
        // logits에서 예측된 토큰 ID 추출
        // Extract predicted token ID from logits
        let nextTokenID = (0..<logits.shape[2].intValue).max {
            logits[[0, predictedTokenIDs.count - 1, $0] as [NSNumber]].floatValue <
                logits[[0, predictedTokenIDs.count - 1, $1] as [NSNumber]].floatValue
        } ?? 0
        
        // 종료 토큰 체크 (예: <EOS> 토큰 ID)
        // Check for end token (e.g., <EOS> token ID)
        if nextTokenID == 3 {
            break
        }
        
        // 예측된 토큰 ID를 추가
        // Add the predicted token ID
        predictedTokenIDs.append(nextTokenID)
        
        // 최대 시퀀스 길이에 도달하면 종료
        // Exit if the maximum sequence length is reached
        if predictedTokenIDs.count >= maxSeqLength {
            break
        }
    }
    
    // 예측된 토큰 ID를 다시 텍스트로 디코딩
    // Decode the predicted token IDs back to text
    let predictedText = tokenizer.decode(tokens: predictedTokenIDs)
    
    // 결과 출력
    // Print the result
    return predictedText
}

@available(macOS, deprecated: 10.14, message: "Use newer API greedyPredict(text:model:tokenizer) async")
@available(iOS, deprecated: 16.0, message: "Use newer API greedyPredict(text:model:tokenizer) async")
@available(tvOS, deprecated: 16.0, message: "Use newer API greedyPredict(text:model:tokenizer) async")
@available(watchOS, deprecated: 9.0, message: "Use newer API greedyPredict(text:model:tokenizer) async")
func greedyPredictDispatch(text: String, model: zenz_v1, tokenizer: Tokenizer, qos: DispatchQoS) async -> String {
    return await withCheckedContinuation { continuation in
        DispatchQueue.global(qos: qos.qosClass).async {
            let result = greedyPredict(text: text, model: model, tokenizer: tokenizer)
            continuation.resume(returning: result)
        }
    }
}

func main() async {
    var benchmarks: [BenchmarkResult] = []
    let model = loadModel()
    guard let model else { fatalError("model not found") }
    let tokenizer = await loadTokenizer()
    guard let tokenizer else { fatalError("tokenizer not found") }
    do {
        // ニホンゴ（Japanese in Katakana Form）→日本語（Japanese in Kanji form）
        if #available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, *) {
            let startAsync = Date()
            let predictedSentenceAsync = await greedyPredict(text: "\u{EE00}ニホンゴ\u{EE01}", model: model, tokenizer: tokenizer)
            print("[Stateless Greedy][Async global][ニホンゴ] output:", predictedSentenceAsync)
            let durationAsync = Date().timeIntervalSince(startAsync)
            print("[Stateless Greedy][Async global][ニホンゴ] duration (s):", durationAsync)
            benchmarks.append(BenchmarkResult(label: "[Stateless Greedy][Async global][ニホンゴ]", duration: durationAsync))
        } else {
            let startAsync = Date()
            let predictedSentenceAsync = await greedyPredictDispatch(text: "\u{EE00}ニホンゴ\u{EE01}", model: model, tokenizer: tokenizer, qos: .userInitiated)
            print("[Stateless Greedy][Async dispatch][ニホンゴ] output:", predictedSentenceAsync)
            let durationAsync = Date().timeIntervalSince(startAsync)
            print("[Stateless Greedy][Async dispatch][ニホンゴ] duration (s):", durationAsync)
            benchmarks.append(BenchmarkResult(label: "[Stateless Greedy][Async dispatch][ニホンゴ]", duration: durationAsync))
        }
        
        let start = Date()
        let predictedSentence = greedyPredict(text: "\u{EE00}ニホンゴ\u{EE01}", model: model, tokenizer: tokenizer)
        print("[Stateless Greedy][Sync main][ニホンゴ] output:", predictedSentence)
        let durationSync = Date().timeIntervalSince(start)
        print("[Stateless Greedy][Sync main][ニホンゴ] duration (s):", durationSync)
        benchmarks.append(BenchmarkResult(label: "[Stateless Greedy][Sync main][ニホンゴ]", duration: durationSync))
    }
    do {
        // カンコクゴヲベンキョウスル（'Study Korean' in Katakana Form）→韓国語を勉強する（'Study Korean' in Kanji form）
        if #available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, *) {
            let startAsync = Date()
            let predictedSentenceAsnyc = await greedyPredict(text: "\u{EE00}カンコクゴヲベンキョウスル\u{EE01}", model: model, tokenizer: tokenizer)
            print("[Stateless Greedy][Async global][カンコクゴ] output:", predictedSentenceAsnyc)
            let durationAsync = Date().timeIntervalSince(startAsync)
            print("[Stateless Greedy][Async global][カンコクゴ] duration (s):", durationAsync)
            benchmarks.append(BenchmarkResult(label: "[Stateless Greedy][Async global][カンコクゴ]", duration: durationAsync))
        } else {
            let startAsync = Date()
            let predictedSentenceAsync = await greedyPredictDispatch(text: "\u{EE00}カンコクゴヲベンキョウスル\u{EE01}", model: model, tokenizer: tokenizer, qos: .userInitiated)
            print("[Stateless Greedy][Async dispatch][カンコクゴ] output:", predictedSentenceAsync)
            let durationAsync = Date().timeIntervalSince(startAsync)
            print("[Stateless Greedy][Async dispatch][カンコクゴ] duration (s):", durationAsync)
            benchmarks.append(BenchmarkResult(label: "[Stateless Greedy][Async dispatch][カンコクゴ]", duration: durationAsync))
        }
        
        let start = Date()
        let predictedSentence = greedyPredict(text: "\u{EE00}カンコクゴヲベンキョウスル\u{EE01}", model: model, tokenizer: tokenizer)
        print("[Stateless Greedy][Sync main][カンコクゴ] output:", predictedSentence)
        let durationSync = Date().timeIntervalSince(start)
        print("[Stateless Greedy][Sync main][カンコクゴ] duration (s):", durationSync)
        benchmarks.append(BenchmarkResult(label: "[Stateless Greedy][Sync main][カンコクゴ]", duration: durationSync))
    }
    
    if #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *) {
        let statefulModel = loadStatefulModel()
        
        guard let statefulModel else { fatalError("Stateful model not found") }
        
        do {
            let start = Date()
            let predictedSentence = greedyPredictStateful(text: "\u{EE00}ニホンゴ\u{EE01}", model: statefulModel, tokenizer: tokenizer)
            print("[Stateful Greedy][Sync main][ニホンゴ] output:", predictedSentence)
            let durationStateful = Date().timeIntervalSince(start)
            print("[Stateful Greedy][Sync main][ニホンゴ] duration (s):", durationStateful)
            benchmarks.append(BenchmarkResult(label: "[Stateful Greedy][Sync main][ニホンゴ]", duration: durationStateful))
        }
        do {
            let start = Date()
            let predictedSentence = greedyPredictStateful(text: "\u{EE00}カンコクゴヲベンキョウスル\u{EE01}", model: statefulModel, tokenizer: tokenizer)
            print("[Stateful Greedy][Sync main][カンコクゴ] output:", predictedSentence)
            let durationStateful = Date().timeIntervalSince(start)
            print("[Stateful Greedy][Sync main][カンコクゴ] duration (s):", durationStateful)
            benchmarks.append(BenchmarkResult(label: "[Stateful Greedy][Sync main][カンコクゴ]", duration: durationStateful))
        }
    }
    print("===== Benchmark Ranking (fast → slow) =====")
    for (index, result) in benchmarks.sorted(by: { $0.duration < $1.duration }).enumerated() {
        print("\(index + 1). \(result.label): \(result.duration) s")
    }
}
