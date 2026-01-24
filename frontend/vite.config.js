import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react-swc";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // Output to the python web folder
    outDir: "../web",
    emptyOutDir: true, // Clears the web folder before building
    rollupOptions: {
      output: {
        // Ensure assets don't get hashed if you reference them statically in python (optional)
        entryFileNames: `assets/[name].js`,
        chunkFileNames: `assets/[name].js`,
        assetFileNames: `assets/[name].[ext]`,
      },
    },
  },
});
