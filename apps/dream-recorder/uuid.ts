/**
 * React Native / Hermes には標準の `crypto.randomUUID` が存在しない場合がある
 * （`Property 'crypto' doesn't exist` の原因）。
 * ここでは expo-crypto の randomUUID を唯一の入口にし、
 * アプリの他の場所からグローバルな `crypto` に直接依存させない。
 */
import * as Crypto from 'expo-crypto';

export function generateId(): string {
  return Crypto.randomUUID();
}
