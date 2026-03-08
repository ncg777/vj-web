/// <reference types="vite/client" />

declare module "*?raw" {
  const content: string;
  export default content;
}

interface FilePickerAcceptType {
  description?: string;
  accept: Record<string, string[]>;
}

interface SaveFilePickerOptions {
  suggestedName?: string;
  types?: FilePickerAcceptType[];
  excludeAcceptAllOption?: boolean;
}

interface Window {
  showSaveFilePicker(options?: SaveFilePickerOptions): Promise<FileSystemFileHandle>;
}

declare module "mp4-muxer" {
  export class FileSystemWritableFileStreamTarget {
    constructor(stream: FileSystemWritableFileStream);
  }

  export interface MuxerOptions {
    target: FileSystemWritableFileStreamTarget;
    video?: {
      codec: string;
      width: number;
      height: number;
    };
    fastStart?: boolean | string;
  }

  export class Muxer {
    constructor(options: MuxerOptions);
    addVideoChunk(chunk: EncodedVideoChunk, meta?: EncodedVideoChunkMetadata): void;
    finalize(): void;
  }
}
