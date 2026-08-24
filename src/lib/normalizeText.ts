// 音声認識テキストの比較用正規化。表示用の生テキスト（rawText・note等）には
// 一切適用しない。全角英数字→半角、大文字化、ひらがな→カタカナ折りたたみを
// 行うことで、「パワステオイル」「ぱわすておいる」等、ASRの表記ゆれを吸収する。
// 比較対象となる別名辞書（登録項目名・判定語エイリアス等）も必ずこの関数で
// 正規化してから比較すること（生テキストにひらがなを含む別名を追加すると、
// 正規化後のテキストと表記が一致せずマッチしなくなるため）。
//
// normalizeChar は必ず1文字→1文字の変換のみ行う（NFKCの多対一分解等、
// 文字数が変わる変換は使わない）。これにより、生テキストから空白だけを
// 取り除いた文字列（buildAligned の display）と比較用文字列（compare）は
// 常に同じ長さ・同じインデックス対応を保つ。この対応関係を使うことで、
// 「マッチング自体はkatakana折りたたみ後のテキストに対して行いつつ、
// 実際にnoteとして表示する内容は折りたたみ前の原文のまま残す」ことができる
// （parseVoiceInspection.ts の TextPair 参照）。
const FULLWIDTH_RE = /[０-９Ａ-Ｚａ-ｚ．]/;
const HIRAGANA_RE = /[ぁ-ゖ]/;

function normalizeChar(ch: string): string {
  if (FULLWIDTH_RE.test(ch)) {
    const code = ch.charCodeAt(0);
    ch = code === 0xff0e ? "." : String.fromCharCode(code - 0xfee0);
  }
  ch = ch.toUpperCase();
  if (HIRAGANA_RE.test(ch)) {
    ch = String.fromCharCode(ch.charCodeAt(0) + 0x60);
  }
  return ch;
}

export interface AlignedText {
  /** マッチング専用の正規化済みテキスト（空白除去・全半角・大小・ひらがなカタカナ統一済み） */
  compare: string;
  /** compareと同じ長さ・同じインデックス対応を持つ、空白のみ除去した原文 */
  display: string;
}

/** rawTextから比較用(compare)と表示用(display)の対を作る。両者は常に同じ長さ。 */
export function buildAligned(rawText: string): AlignedText {
  let compare = "";
  let display = "";
  for (const ch of rawText) {
    if (/\s/.test(ch)) continue;
    display += ch;
    compare += normalizeChar(ch);
  }
  return { compare, display };
}

/** 比較専用の正規化済みテキストのみが必要な場合（別名辞書の正規化等）に使う。 */
export function normalize(text: string): string {
  return buildAligned(text).compare;
}
