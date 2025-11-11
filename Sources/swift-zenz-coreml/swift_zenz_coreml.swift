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

// Greedy search를 사용하여 예측 수행
// Perform prediction using Greedy search
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)
func greedyPredictStateful(text: String, model: zenz_v1_stateful, tokenizer: Tokenizer) -> String {
    let state = model.makeState()
    
    var inputIDs = tokenizer.encode(text: text)
    print("[Stateful Greedy] inputIDs:", text, inputIDs)
    
    let maxSeqLength = 128
    let batchSize = 1
    var predictedTokenIDs = inputIDs
    
    while true {
        // 입력 / attention mask는 이미 Int32 잘 쓰고 있음
        let inputArray = try? MLMultiArray(
            shape: [NSNumber(value: batchSize), NSNumber(value: predictedTokenIDs.count)],
            dataType: .int32
        )
        for (index, token) in predictedTokenIDs.enumerated() {
            inputArray?[index] = NSNumber(value: token)
        }
        
        let attentionMask = try? MLMultiArray(
            shape: [NSNumber(value: batchSize), NSNumber(value: predictedTokenIDs.count)],
            dataType: .int32
        )
        for i in 0..<predictedTokenIDs.count {
            attentionMask?[i] = 1
        }
        
        guard let inputArray, let attentionMask else { return "" }
        
        let input = zenz_v1_statefulInput(input_ids: inputArray, attention_mask: attentionMask)
        
        guard let output = try? model.prediction(input: input, using: state) else { return "" }
        
        // logits로 변경
        let logits = output.logits
        
        let nextTokenID = (0..<logits.shape[2].intValue).max {
            logits[[0, predictedTokenIDs.count - 1, $0] as [NSNumber]].floatValue <
            logits[[0, predictedTokenIDs.count - 1, $1] as [NSNumber]].floatValue
        } ?? 0
        
        if nextTokenID == 3 {
            break
        }
        
        predictedTokenIDs.append(nextTokenID)
        
        if predictedTokenIDs.count >= maxSeqLength {
            break
        }
    }
    
    let predictedText = tokenizer.decode(tokens: predictedTokenIDs)
    return predictedText
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
