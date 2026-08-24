import { statusLabel } from "../lib/format";
import type { ParsedUnmatched } from "../lib/parseVoiceInspection";

interface UnmatchedConfirmModalProps {
  pendingUnmatched: ParsedUnmatched;
  pendingUnmatchedCount: number;
  onConfirmAdd: () => void;
  onDiscard: () => void;
}

/**
 * 未登録項目の確認UI。position:fixedのオーバーレイとして画面に重ねて表示し、
 * 通常の書類の流れ（document flow）の外に置く。これにより、この確認UIの
 * 有無や内容量が「全項目一覧」の表示位置に一切影響しない。
 */
export function UnmatchedConfirmModal({
  pendingUnmatched,
  pendingUnmatchedCount,
  onConfirmAdd,
  onDiscard,
}: UnmatchedConfirmModalProps) {
  return (
    <div className="modal-overlay">
      <div className="modal-panel">
        <div className="modal-section-title">
          未登録項目の可能性{pendingUnmatchedCount > 1 ? `（ほか${pendingUnmatchedCount - 1}件あります）` : ""}
        </div>
        <div className="recognition-result-unmatched-name">{pendingUnmatched.customCategoryName}</div>
        {pendingUnmatched.status && <div>判定: {statusLabel(pendingUnmatched.status)}</div>}
        <div className="recognition-result-actions">
          <button type="button" className="big-button big-button--primary" onClick={onConfirmAdd}>
            未登録項目として追加
          </button>
          <button type="button" className="big-button" onClick={onDiscard}>
            追加しない
          </button>
        </div>
      </div>
    </div>
  );
}
