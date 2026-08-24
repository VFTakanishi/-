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

/**
 * 同じ長さの2文字列を「位置ごとに」比較し、文字が異なる箇所の数を返す。
 *
 * 項目名のfuzzy matchでは、長さが同じ窓同士の比較にLevenshtein距離ではなく
 * こちらを使う。Levenshteinは挿入・削除も許すため、例えば本物の完全一致の
 * 直前の窓（先頭を1文字捨てて末尾に1文字補う「1文字ずれ」）が常に距離2に
 * なってしまい、距離2までの許容と組み合わせると隣接語まで誤って
 * fuzzy一致してしまう。Hamming距離は「ずれ」に低いスコアを与えないため、
 * 本当に近い（同じ位置の文字が数個違うだけの）誤認識だけを安全に拾える。
 * 長さが異なる場合はfuzzy match対象外として扱うため呼び出し側で長さを揃える。
 */
export function hammingDistance(a: string, b: string): number {
  if (a.length !== b.length) return Math.max(a.length, b.length);
  let distance = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) distance++;
  }
  return distance;
}
