import { Directory, File, Paths } from 'expo-file-system';
import type { FileInfo, FileStorageService, PersistedFileEntry } from './types';

const RECORDINGS_DIR_NAME = 'dream-recordings';

/**
 * expo-file-system (新API: File / Directory / Paths) を使った永続保存実装。
 * 録音は expo-audio によっていったんキャッシュ領域(directory: 'cache')に作られるため、
 * ここで Documents 配下の専用フォルダへコピーし、システムに消されにくくする。
 */
export class ExpoFileStorageService implements FileStorageService {
  private readonly dir = new Directory(Paths.document, RECORDINGS_DIR_NAME);

  async ensureDirectoryExists(): Promise<void> {
    if (!this.dir.exists) {
      this.dir.create({ intermediates: true, idempotent: true });
    }
  }

  async copyToPersistent(tempUri: string, destFileName: string): Promise<string> {
    await this.ensureDirectoryExists();
    const sourceFile = new File(tempUri);
    const destFile = new File(this.dir, destFileName);
    await sourceFile.copy(destFile, { overwrite: false });
    return destFile.uri;
  }

  async getFileInfo(uri: string): Promise<FileInfo> {
    const file = new File(uri);
    if (!file.exists) {
      return { exists: false, size: 0 };
    }
    return { exists: true, size: file.size };
  }

  async listPersistedFiles(): Promise<PersistedFileEntry[]> {
    await this.ensureDirectoryExists();
    const entries = this.dir.list();
    const files: PersistedFileEntry[] = [];
    for (const entry of entries) {
      if (entry instanceof File) {
        files.push({ uri: entry.uri, fileName: entry.name, size: entry.exists ? entry.size : 0 });
      }
    }
    return files;
  }
}
