import { useState } from "react";
import { InspectionScreen } from "./screens/InspectionScreen";
import { StartScreen } from "./screens/StartScreen";
import { SummaryScreen } from "./screens/SummaryScreen";
import { getInspection } from "./lib/storage";
import { trackAnalyticsEvent } from "./lib/analytics";

type Screen = "start" | "inspection" | "summary";

export default function App() {
  const [screen, setScreen] = useState<Screen>("start");
  const [inspectionId, setInspectionId] = useState<string | null>(null);

  const openInspection = (id: string) => {
    setInspectionId(id);
    setScreen("inspection");
  };

  const backToStart = () => {
    trackAnalyticsEvent("back_to_top");
    setScreen("start");
  };

  if (screen === "start" || !inspectionId) {
    return <StartScreen onOpenInspection={openInspection} />;
  }

  if (screen === "summary") {
    const inspection = getInspection(inspectionId);
    if (!inspection) {
      setScreen("start");
      return null;
    }
    return (
      <SummaryScreen
        inspection={inspection}
        onBack={() => setScreen("inspection")}
        onBackToStart={backToStart}
      />
    );
  }

  return (
    <InspectionScreen
      inspectionId={inspectionId}
      onOpenSummary={() => setScreen("summary")}
      onBackToStart={backToStart}
    />
  );
}
