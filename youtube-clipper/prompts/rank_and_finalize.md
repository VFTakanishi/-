# Stage 2: hook seed coverage方式による最終編集設計（final edit design）

以下は、ポッドキャスト全体を複数のチャンクに分けてそれぞれから抽出した発見内容です（チャンクの境界は約1分オーバーラップさせて重複抽出しているため、同じ発言区間を指す発見が複数含まれている場合があります）。全文の文字起こしは渡されていません — 各項目が実際に使う発言のsegment情報だけで判断してください。3つの入力グループがあります:

- `coverage_targets`: Python側が選んだ、試作すべきhook_seed一覧（強い引きになり得る発話の候補）
- `support_materials`: 完成candidateを組み立てるための部品（reason/example/context/payoff）
- `fallback_spans`: 成果物0件を避けるための安全な保険素材

**あなたの仕事は、`coverage_targets`の各hook_seedについて、必ず1件のattempt（試作結果）を返すことです。** 試作できるなら完成candidateとして、試作できないなら理由付きの明示的なrejectとして、どちらかを必ず返してください。**無言でhook_seedを無視して終了することは禁止です。**

タイトル・説明文・煽り文などの文章は一切書きません。渡された一覧に存在する`segment_id`だけを使い、後述の出力形式で結果を返してください。

## 各入力グループの情報

`coverage_targets`（各要素）:
- `hook_seed_id`: このhook_seedを指す識別子。**あなたの`attempts`出力で、必ずこのidを使って対応するattemptを1件返してください。**
- `signal_type`: このhook_seedが強いと判断された理由（参考情報）
- `soft_score`: Stage1の自己採点（参考情報。hard rejectには使われていません）
- `segments`: `start_segment_id`/`end_segment_id`/`text`/`start_sec`/`end_sec`のリスト
- `lookahead`: このhook_seedの最後のsegmentより後に実際に続く発言（参考用の閲覧材料）

`support_materials`（各要素）:
- `material_id`: 識別子（あなた自身の出力では使いません）
- `material_type`: `reason` / `example` / `context` / `payoff`
- `usefulness_score`: その素材自身の目的における有用性スコア（参考情報）
- `segments`
- `lookahead`

`fallback_spans`（各要素）:
- `fallback_span_id`: 識別子（fallback candidateを設計する際の元ネタとして参照する用途。あなた自身の出力する`fallback_id`とは別物です）
- `safety_score`: Stage1の自己採点（参考情報）
- `segments`
- `lookahead`

## attemptの設計（coverage必須）

**`coverage_targets`がN件なら、`attempts`もちょうどN件返してください。** 各hook_seedについて、以下のどちらかを選びます:

- `status: "candidate"`: そのhook_seedを起点に完成candidateを設計できた場合。`candidate`フィールドに設計内容を入れる。
- `status: "rejected"`: どう編集しても実用candidateにならないと判断した場合。`reject_reason_code`に理由コードを入れる（`candidate`は入れない）。

**重要（今回の核心）**: 強いhook_seed（例: 具体的な金額・損失を示すもの）は、**必ず一度は完成candidateとして試作してください。** 一位にする必要はありませんが、一度も試さずに`rejected`にすることは禁止です。「弱いhook_seedだけを使い、近くにあるもっと強いhook_seedを一度も試さずに終わる」構造を避けるための仕組みです。

### hook_seed単体では対象が不明な場合（自己完結させる）

hook_seed自体が、例えば「壊れた場合の損失が大きい」とだけ述べていて、「何が壊れるのか」という対象が不足している場合があります。この場合、作文はせず、以下を使って対象を補ってください:

- 直前/直後の実発話（lookahead含む）
- 別の`support_material`
- `start_anchor_text`/`end_anchor_text`
- segmentの並び替え

例（構造の例）: hook_seedが「対象不明の強い損失の言及」だけの場合、近くの実発話に「何についての話か」を明示する一言（例: 特定の部品名を挙げる発話）があれば、それを`context`として先頭に自然に繋げられないか検討してください。ただし不自然な接続は禁止です（下記「カット接続の自然さ」参照）。自己完結させられない場合は、`reject_reason_code: "insufficient_context_available"`でrejectしてください。

### `reject_reason_code`（`status: "rejected"`の場合に必須）

- `insufficient_context_available`: 対象・文脈を実発話で補えず、自己完結させられない
- `unnatural_junction_only`: どう組んでもsegment同士のつなぎ目が不自然になる
- `duration_infeasible`: 50秒以内で自然に再構成しても収まらない、または20秒に届かない
- `disfluency_unavoidable`: 言い淀み・言い直しを避けた構成が作れない
- `semantic_closure_unavailable`: hookの問い・主張を回収する実発話が見つからない
- `duplicate_of_stronger_attempt`: 別のより強いattemptと実質的に同じ構成になってしまう

## 完成candidateの設計手順（attempt・fallback共通）

1. **hook segmentを決める**: attemptの場合はそのhook_seedの実発話（自己完結のため補完してよい）。fallback candidateの場合はfallback_spanの実発話。
2. **hookを回収するreason/answerをsupport_materialsから探す**: 必要ならsupport_materials/他のhook_seed/fallback_spanのsegmentを組み合わせてよい。
3. **必要なcontext/example/payoffを追加する**。
4. **終了候補地点より後ろの`lookahead`も確認する**。
5. **意味的に自然な終了地点を選び、20〜50秒に収める**: 詳細は下記「終了地点の設計」。
6. 50秒を超える場合は同じ出力内で再構成する（下記参照）。

**厳守事項（すべて必ず守ること）:**

- 実在する`segment_id`の組み合わせのみを使うこと。存在しない`segment_id`を作文しないこと。
- 発話を作文・要約・言い換えしないこと。実際に流れるテキストは常に実発話そのものでなければならない。
- 同一テーマ内での組み合わせに限ること（無関係な話題のsegmentを無理につなげない）。
- 文中の単語の並べ替えは禁止。segment/anchor単位の並べ替え・組み合わせのみ。

## segment数について（1〜3推奨・最大6許容）

**目的は「segment数を減らすこと」ではなく「投稿できる質の高いShortsを作ること」です。** 可能なら1〜3個のsegmentで自然に組めるものを優先してください。ただし、hookを回収する理由・具体例・payoffまで含めて自然につなげるために4個以上必要なら、4〜6個のsegmentを普通に使ってください。segment数が多いという理由だけでcandidateを避けたり、rejectedにしたりしないでください。

逆に、無理に1〜3個へ押し込んで以下のような弊害が出るなら、それは避けてください:
- 必要な理由・文脈を削ったせいで`hook_claim_resolved`が満たせなくなる
- 接続を無理に詰めたせいで「カット接続の自然さ」が崩れる
- 意味が薄くなる、または話が唐突に終わる

4〜6個のsegmentを使う場合は、以下を厳しく確認してください:
- 各cutが本当に必要か（削っても意味が保たれるなら削ること）
- 同じ内容を繰り返すだけのcutが混ざっていないか
- context/exampleを不必要に細かく分割していないか
- 隣接segment同士の接続がすべて自然か（「カット接続の自然さ」は2segmentの場合と同じ基準で、増えた分だけ全てのつなぎ目に適用されます）
- 聞いていてブツ切れ感・テンポの悪化が出ないか

7個以上のsegmentは常に禁止です。どうしても7個以上必要になる場合は、そのcandidateを設計しないでください（`status: "rejected"`の場合は`reject_reason_code`を適切なものにしてください）。

**評価すべきはsegment数そのものではなく、接続の自然さ・意味的完結・テンポです。** 2segmentでも接続が不自然・意味が破綻していればreject対象であり、6segmentでも自然に繋がっていればaccept対象です。

## 冒頭の強さ（`opening_hook_strength`、相対比較用のsoft score）

**この数値はhard rejectには使われません。** 相対比較（下記「相対比較（ranking）」）の一材料として、自分が設計した完成candidateの最初のsegmentの実テキストを厳密に自己採点してください:
- 強い主張・明確な断言、常識と逆の結論
- 意外な事実、具体的な数字、明確な比較
- 故障・失敗・損失など強い結果
- 明確な疑問、視聴者への直接的な問い
- 結論先出し、聞いた瞬間に問題や結論がわかる一言

自己採点は厳しく行うこと（甘い採点を禁止）。ただし数値の高さだけでcandidateの採否が決まるわけではありません（下記`opening_self_contained`が本質的なhard gateです）。

## `opening_self_contained`の判定（hard gate）

**冒頭は少なくとも、「初見の視聴者が、前の動画を見ていなくても何について何を言っているか理解できる」ことが必須です。** 「これ」「それ」「その場合」等の指示語で始まっていなくても、対象・主語が実質的に不明であれば`opening_self_contained: false`にしてください。

実例（構造の例）: 「(対象不明)何かがちょくちょく壊れる、とだけ述べていて、何が壊れるのか冒頭だけでは分からない発話」→ 指示語で始まっていなくても`opening_self_contained: false`。対象を実発話で補えた場合のみ`true`にしてよい。

**迷ったら`false`にしてください。** `opening_self_contained: false`のcandidateはprimaryとして採用されません（Python側のhard gate）。

## カット接続の自然さも独立して評価すること

各segment単体・冒頭・末尾が良くても、**segment同士のつなぎ目が不自然なら設計として不適格です。** 自分が設計した`segments`を上から順に実際に読み上げたと仮定し、隣接するsegment同士のつながりが日本語として自然かを確認してください。

特に以下のパターンは、その組み合わせでの設計を避けてください:
- あるsegmentが「〜のであれば」「〜なら」「〜たら」「〜れば」「〜ので」「〜けど」「〜けども」等、後続を要求する表現で終わっているのに、次のsegmentが全く別の条件文・話題から始まっている（例: 「車を冷やしますっていうのであれば」の次が「連続周回をする場合は」のような、無関係な条件へ飛ぶ接続）
- 最後のsegmentが発話途中で終わっている（「〜良いかもしれないんですけども」のような継続表現で終わっている）

## `hook_claim_resolved`の判定（意味的な完結性、hard gate）

自分が設計したcandidateの最初のsegment（hook）が以下のいずれかを提示している場合:

- A. 明確な問い
- B. 「実は〜」型の意外な主張
- C. 明確な比較
- D. 原因・理由を知りたくなる主張（例:「Xの方がYより効率が良い」）
- E. open loop（意図的に核心を伏せる構成）

その場合、後半（`context`/`answer`/`payoff`のsegment）に、**その問い・主張を実際に回収する発話が存在すること**を確認し、`hook_claim_resolved: true`にしてください。回収する発話が見つからない場合は`hook_claim_resolved: false`にするか、**そもそもそのcandidateを設計しないでください**（存在しない理由を作文して埋めるのは禁止）。以下のような内容では回収したことになりません:

- 主張の単純な繰り返し
- 「〜だと思います」「〜な気がします」のような感想・推測だけ
- hookの主張と直接関係のない一般論・別論点
- 結論を言うだけで理由を説明しない

**実例（reject対象）**:
hook「ギアを入れてアクセルオフの方がニュートラルより燃費は良いです」に対し、bodyが「〜は推奨されていません」という安全上の注意と「Nレンジの方が燃費が良い気がします」という感覚論だけで終わっている場合 → hookが提示した「なぜアクセルオフの方が燃費が良いのか」という理由に一切触れていないため、`hook_claim_resolved: false`。

**実例（accept対象）**: 同じhookに対し、**別のsupport_materialの中に**理由・仕組みを説明する実発話が存在する場合 → そのsegmentをanswer/payoffとして組み合わせれば`hook_claim_resolved: true`になります。

**重要**: これは一般ルールです。特定のテーマの専門用語を合否基準として固定しないでください。あなたが判断するのは「hookが作った問い・期待に、この設計の実発話が答えているか」だけです。

## 終了地点の設計（意味的完結性優先）— 「尺が合うか」ではなく「話が終わったと感じるか」

**重要な前提**: 「23秒だから短い」「40秒あるから大丈夫」という尺の長さそのものは問題ではありません。23秒でも意味的にきれいに完結していれば合格、40秒あっても話の途中では不合格です。今回あなたに求められているのは、「尺条件を満たす最初の地点で切る」のではなく、「このcandidate単体で視聴者が『話が終わった』と自然に感じる地点」を能動的に選ぶことです。

**最短で終われる地点ではなく、最適な自然終了を選ぶこと。** 「なるべく早く終われ」という指示ではありません。逆に「長く使え」でもありません。

終了地点を選ぶときの基準（すべて確認すること）:
- A. hookで作った期待（問い・主張・比較・原因）が回収されているか
- B. 理由・説明が言いかけで終わっていないか
- C. 最後の発話だけを聞いても、明らかに「まだ続く」と感じさせないか
- D. 締めくくりとして自然に聞こえるか
- E. その後の発言が単なる蛇足・脱線・別の具体例の追加なら、そこで止めてよい（無理に含めない）

**具体例**: 23秒地点=理由の説明がまだ途中、29秒地点=理由は言えたがまだ続いている、34秒地点=意味的にきれいに完結、41秒地点=追加の具体例で蛇足、という場合 → **34秒を選ぶこと**（23秒でも41秒でもない）。

**`lookahead`の使い方**: 各入力の`lookahead`は、その最後のsegmentより後に実際に続く発言です。今使おうとしている終了地点で本当に良いか、`lookahead`の中にもっと自然な締めがないかを確認してください。ただし`lookahead`にある発言を必ず使わなければならないわけではありません。`lookahead`が別の話題に移っている場合は、無理に含めないでください。

**設計手順（この順序で考えること）**:
1. 強いhookを選ぶ
2. hookを回収するreason/answerを選ぶ
3. 必要なcontext/example/payoffを追加する
4. 終了候補地点より後ろの`lookahead`も確認する
5. 意味的に自然な終了地点を決める
6. 総尺を計算する
7. 50秒以内に収まっていれば、そのままcandidateとして確定する
8. 50秒を超えていれば、次の「50秒を超える場合の再構成」に従って作り直す
9. 作り直した後、再度意味的完結性（A〜E）を確認する
10. 自然に完結していて20〜50秒に収まっていれば、candidateとして確定する

### 50秒を超える場合の再構成（即rejectしない）

自然な構成が50秒を超えても、**すぐにreject/50秒地点で機械的に切る、のどちらも禁止です。** まず、意味を保ったまま50秒以内に収まる自然な構成を試みてください。

削る優先順位（この順で検討する）:
1. 冗長な`context`
2. 重複している説明
3. 補助的な`example`
4. 意味の理解に必須ではない`payoff`

できる限り維持すること: hook、hookを回収するanswer、必要な理由・結論。

**許可されること**: `context`/`example`/重複説明のsegmentを丸ごと外す、同じ意味を保てる別の実発話segmentに差し替える、`start_anchor_text`/`end_anchor_text`で実発話の自然な意味単位に切り詰める。

**禁止されること**: 50秒ちょうどで機械的に打ち切る、文の途中で切る、segmentの末尾を意味を無視して適当に切る、実際にない発話を作文・要約・言い換えする。

この再構成は**追加のAPI呼び出しを発生させません** — あなたは今のこの1回の出力の中で、最初の自然な構成が50秒を超えると分かった時点で、その場で短い完成構成へ作り直してから出力してください。再構成してもなお、hook・hookを回収するanswer・意味的完結性を維持したまま50秒以内に収められない場合に限り、そのcandidateの設計自体を諦めてください（attemptなら`reject_reason_code: "duration_infeasible"`）。

## `start_anchor_text` / `end_anchor_text`（segmentの一部だけを使う）

segmentの全文を頭から末尾まで使う必要はありません。任意で`start_anchor_text`（segmentの実際の開始位置を動かす）・`end_anchor_text`（segmentの実際の終了位置を動かす）を指定できます。

- 必ずそのsegmentの実テキストに**実在する、連続した実発話の一部**であること。作文・要約・言い換えは禁止。
- 単語の途中で始まる/終わる位置は指定できない（word境界に一致する必要がある。一致しない場合はプログラム側が自動的に無効化し、segmentの元の範囲にfallbackする — 推測でのcutは行われない）。
- 短い句・節（数単語程度）を想定している。segmentの内容を丸ごと書き写す必要はない。
- 不要なら省略してよい。
- `start_anchor_text`は弱い前置きを削る用途（例: segment全文が「よくある話が、私も乗っているZN6-86であったり」なら`start_anchor_text: "ZN6-86であったり"`で「よくある話が」を削る）。
- `end_anchor_text`は末尾の不要な補足・重複説明を削って尺を収める用途（例: segment全文が「その理由は減速時の燃料カットが働くからです。以上が本日の内容でした。」で、後半の「以上が本日の内容でした。」が不要なら`end_anchor_text: "その理由は減速時の燃料カットが働くからです。"`）。

## fallback_candidatesの設計

`fallback_spans`は成果物0件を避けるための安全な保険素材です。主に`fallback_spans`から、必要なら`support_materials`と組み合わせて、fallback candidateを設計してください（最大件数は渡された枠の分だけ）。

**fallbackだからといって基準を緩めないでください。** 意味的完結性（`hook_claim_resolved`）・カット接続の自然さ・尺・`opening_self_contained`・意味的な終了地点設計のすべてを、primaryのattemptと全く同じ基準で満たすように設計してください。満たせないなら、そのfallback candidateは0件のままにしてください（無理に出力しない）。

各fallback candidateには、あなた自身が短いid（例: `"fb1"`, `"fb2"`）を`fallback_id`として付けてください。

## 相対比較（ranking）

`attempts`のうち`status: "candidate"`になったもの、および`fallback_candidates`のすべてを対象に、以下の観点で相対的に比較し、強い順にidを並べた`ranking`を返してください（`status: "rejected"`のattemptは対象外）:

- opening self-containment（`opening_self_contained`）
- attention power（引きの強さ）
- specificity（具体性、数字の有無）
- surprise（意外性）
- clarity（分かりやすさ）
- pacing（テンポ）
- information density（情報密度）
- semantic satisfaction（意味的な満足度）
- overall publishability（総合的な投稿適性）

`ranking`の各要素は、attemptから来たものなら対応する`hook_seed_id`、fallback candidateから来たものなら`fallback_id`を使ってください。長い理由文は不要です。id列だけを返してください。

## 出力

- `attempts`: `coverage_targets`と同数。各要素は`hook_seed_id`/`status`（`candidate` or `rejected`）/ `status: candidate`の場合のみ`candidate`（下記フィールド）/ `status: rejected`の場合のみ`reject_reason_code`
- `fallback_candidates`: 各要素は`fallback_id`/`candidate`（下記フィールド）
- `ranking`: 上記「相対比較」参照

`candidate`（attempt・fallback共通のフィールド）:
- `hook_type`: `open_loop` / `strong_take` / `surprising_fact` / `story`
- `segments`: このcandidateが実際に使うsegmentのリスト（**並び順が実際の再生順**、1〜6個。1〜3個推奨、詳細は上記「segment数について」）。各要素は`role`（`hook`/`context`/`answer`/`payoff`。最初のsegmentは必ず`hook`）、`start_segment_id`/`end_segment_id`（実在するsegment ID、inclusive）、任意で`start_anchor_text`/`end_anchor_text`
- `opening_hook_strength`: 0〜100（soft score、上記「冒頭の強さ」参照）
- `score`: 0〜100の総合スコア（soft score、フックの強さ・単体での満足度・本編への興味喚起のバランスで評価）
- `opening_self_contained`: 上記「`opening_self_contained`の判定」参照（hard gate）
- `hook_claim_resolved`: 上記「`hook_claim_resolved`の判定」参照（hard gate）
- `semantic_ending_complete` / `ending_rationale_code` / `recomposed_for_duration`: 上記「終了地点の設計」参照

すべてのcandidateで、これらのフィールドすべてに必ず値を設定してください。
