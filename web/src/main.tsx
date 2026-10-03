import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Spinner, ToastProvider } from "./components/ui";
import { ThemeProvider } from "./lib/theme";
import "./styles.css";

const page = (load: () => Promise<{ default: React.ComponentType }>) => {
  const C = lazy(load);
  return (
    <Suspense fallback={<div className="py-24 grid place-items-center"><Spinner label="Loading…" /></div>}>
      <C />
    </Suspense>
  );
};

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: page(() => import("./pages/Overview")) },
      { path: "/physiology/meal", element: page(() => import("./pages/MealLab")) },
      { path: "/physiology/beta-cell", element: page(() => import("./pages/BetaCell")) },
      { path: "/physiology/minimal-model", element: page(() => import("./pages/MinimalModel")) },
      { path: "/physiology/clinical", element: page(() => import("./pages/Clinical")) },
      { path: "/physiology/biophysics", element: page(() => import("./pages/Biophysics")) },
      { path: "/cells/explorer", element: page(() => import("./pages/CellExplorer")) },
      { path: "/cells", element: page(() => import("./pages/CellTheatre")) },
      { path: "/dna", element: page(() => import("./pages/DnaLab")) },
      { path: "/rna/sequences", element: page(() => import("./pages/RnaSequences")) },
      { path: "/rna/predict", element: page(() => import("./pages/RnaPredict")) },
      { path: "/rna/structures", element: page(() => import("./pages/StructureViewer")) },
      { path: "/data", element: page(() => import("./pages/DatasetExplorer")) },
      { path: "/risk", element: page(() => import("./pages/RiskModel")) },
      { path: "/assistant", element: page(() => import("./pages/Assistant")) },
      { path: "/references", element: page(() => import("./pages/References")) },
      { path: "*", element: page(() => import("./pages/NotFound")) },
    ],
  },
]);

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 30_000 } } });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={qc}>
        <ToastProvider>
          <RouterProvider router={router} />
        </ToastProvider>
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>,
);
