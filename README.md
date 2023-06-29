クラス構成

UIコントローラ (UIController): ユーザインターフェース(UI)の制御を行い，ユーザとの対話を処理する役割をもつ。
入力処理クラス (InputProcessor): ユーザからの入力を受け取り，有効性を確認し，必要に応じて変換や整形を行い最終結果を返す役割をもつ。
メッセージ表示クラス (MessageDisplay): ユーザへのメッセージの管理と表示を担当し，情報やエラーメッセージなどのメッセージを適切に処理する役割をもつ。
数値検証クラス (NumericValidator): 与えられた数値が有効値かどうかを判定し，無効値の場合には補正してメッセージを表示し，有効値を返す役割をもつ。
選択肢判定クラス (OptionEvaluator): 与えられた数値が2つの選択肢のどちらを表すかを判定し，結果を返す役割をもつ。
処理フローコントローラ (ProcessFlowController): 処理のフローを制御し，処理の継続または終了を確認し，処理させる役割をもつ。
暦計算コントロールクラス (CalendarControl): 暦計算の進行を管理し，日付計算クラスを呼び出して計算を行う役割をもつ。
日付計算クラス (DateCalculator): 与えられた日付データを利用して，日付の計算を行い，計算結果を返す役割をもつ。
通信ハンドラクラス (CommunicationHandler): TCP/IPプロトコルを使用して通信の準備，通信終了，データの送受信を行う役割をもつ。

クラス図
@startuml
class UIController {
  - inputProcessor: InputProcessor
  - messageDisplay: MessageDisplay
  - processFlowController: ProcessFlowController
  - communicationHandler: CommunicationHandler
}

class InputProcessor {
  - messageDisplay: MessageDisplay
  - numericValidator: NumericValidator
  - optionEvaluator: OptionEvaluator
}

class MessageDisplay
class NumericValidator
class OptionEvaluator

class ProcessFlowController {
  - messageDisplay: MessageDisplay
}

class CalendarControl {
  - dateCalculator: DateCalculator
  - communicationHandler: CommunicationHandler
}

class DateCalculator
class CommunicationHandler

UIController --> InputProcessor
UIController --> MessageDisplay
UIController --> ProcessFlowController
UIController --> CommunicationHandler
InputProcessor --> NumericValidator
InputProcessor --> OptionEvaluator
InputProcessor --> MessageDisplay
ProcessFlowController --> MessageDisplay
CalendarControl --> DateCalculator
CalendarControl --> CommunicationHandler
@enduml

開始日入力シーケンス図
@startuml
actor User
participant UIController
participant InputProcessor
participant MessageDisplay
participant NumericValidator

UIController -> InputProcessor: requestStartDateInput()
activate InputProcessor
InputProcessor -> MessageDisplay: requestInputMessage()
activate MessageDisplay
MessageDisplay -> User: message
deactivate MessageDisplay
activate User
User --> InputProcessor : userInput
deactivate User
InputProcessor -> NumericValidator: validate(input)
activate NumericValidator

opt isUncorrectedValue
NumericValidator -> NumericValidator: adjustToCorrectedValue(input)
NumericValidator -> MessageDisplay: requestCorrectionMessage()
activate MessageDisplay
MessageDisplay -> User: message
deactivate MessageDisplay
end

NumericValidator -> InputProcessor: correctedValue(correctedValue)
deactivate NumericValidator
InputProcessor -> UIController: validInput(correctedValue)
deactivate InputProcessor
@enduml
