import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        felt: {
          950: "#061d1a",
          900: "#082b26",
          800: "#0d3b34",
          700: "#145247"
        },
        gold: {
          300: "#f8d777",
          400: "#e9b949",
          500: "#c99124"
        },
        paper: "#f8f2e8",
        ink: "#15231f"
      },
      fontFamily: {
        sans: ["Inter", "\"Noto Sans JP\"", "system-ui", "sans-serif"],
        display: ["Georgia", "\"Yu Mincho\"", "serif"]
      },
      boxShadow: {
        card: "0 6px 18px rgba(0, 0, 0, 0.22)",
        gold: "0 0 0 1px rgba(233,185,73,.45), 0 8px 28px rgba(0,0,0,.22)"
      }
    }
  },
  plugins: []
} satisfies Config;
