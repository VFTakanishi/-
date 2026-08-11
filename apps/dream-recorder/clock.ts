export interface Clock {
  now(): number;
  nowIso(): string;
}

export class SystemClock implements Clock {
  now(): number {
    return Date.now();
  }
  nowIso(): string {
    return new Date().toISOString();
  }
}
