export type FileInfo = {
  exists: boolean;
  size: number;
};

export type PersistedFileEntry = {
  uri: string;
  fileName: string;
  size: number;
};

export interface FileStorageService {
  /** 永続録音フォルダが無ければ作成する。 */
  ensureDirectoryExists(): Promise<void>;
  /** 一時URIのファイルを永続領域へコピーし、最終URIを返す。 */
  copyToPersistent(tempUri: string, destFileName: string): Promise<string>;
  /** ファイルの存在とサイズを確認する。 */
  getFileInfo(uri: string): Promise<FileInfo>;
  /** 永続録音フォルダ内の全ファイルを列挙する（整合性検査・孤立ファイル検出用）。 */
  listPersistedFiles(): Promise<PersistedFileEntry[]>;
}
