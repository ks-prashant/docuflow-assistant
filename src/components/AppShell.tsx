import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";

function NavLink({ to, label }: { to: string; label: string }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const active = pathname === to;
  return (
    <Link
      to={to}
      className={
        "px-3 py-1.5 text-sm rounded-full transition-colors " +
        (active
          ? "bg-foreground text-background"
          : "text-muted-foreground hover:text-foreground")
      }
    >
      {label}
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-border/60 bg-background/80 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto max-w-5xl px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2.5">
            <div className="h-7 w-7 rounded-lg bg-primary flex items-center justify-center">
              <div className="h-2.5 w-2.5 rounded-sm bg-primary-foreground" />
            </div>
            <div className="flex items-baseline gap-1.5 leading-none">
              <span className="font-display text-xl">DocuFlow</span>
              <span className="text-[11px] text-muted-foreground/80 font-normal">
                by Prashant
              </span>
            </div>
          </Link>
          <nav className="flex items-center gap-1">
            <NavLink to="/" label="Documents" />
            <NavLink to="/chat" label="Chat" />
          </nav>
        </div>
      </header>
      <main className="flex-1">{children}</main>
    </div>
  );
}
