import React from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { DreamStoreProvider } from '../src/state/DreamStoreContext';
import { theme } from '../src/ui/theme';

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <DreamStoreProvider>
        <StatusBar style="light" />
        <Stack
          screenOptions={{
            headerShown: false,
            contentStyle: { backgroundColor: theme.background },
            animation: 'fade',
          }}
        />
      </DreamStoreProvider>
    </SafeAreaProvider>
  );
}
