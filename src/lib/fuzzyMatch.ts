/**
 * 2文字列間の編集距離（挿入・削除・置換）を計算する。
 * 項目名の軽微な音声誤認識を安全に吸収するためだけに使う、小さな汎用ユーティリティ。
 */
export function levenshteinDistance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  let prev = new Array(b.length + 1);
  let curr = new Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;

  for (let i = 1; i <= a.length; i++) {
    curr[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        prev[j] + 1, // 削除
        curr[j - 1] + 1, // 挿入
        prev[j - 1] + cost // 置換 / 一致
      );
    }
    [prev, curr] = [curr, prev];
  }

  return prev[b.length];
}
