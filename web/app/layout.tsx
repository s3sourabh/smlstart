import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "slm-125m Playground",
  description: "Text-completion playground for the 125M legal/financial SLM",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
