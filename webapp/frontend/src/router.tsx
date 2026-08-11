import {
  createContext,
  type AnchorHTMLAttributes,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";

interface RouterContextValue {
  pathname: string;
  navigate: (to: string, options?: { replace?: boolean }) => void;
}

const RouterContext = createContext<RouterContextValue | null>(null);
const ParamsContext = createContext<Record<string, string>>({});

function currentPath(): string {
  const raw = window.location.hash.replace(/^#/, "");
  return raw.startsWith("/") ? raw : "/";
}

export function HashRouter({ children }: { children: ReactNode }) {
  const [pathname, setPathname] = useState(currentPath);

  useEffect(() => {
    const update = () => setPathname(currentPath());
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);

  const navigate = useCallback((to: string, options?: { replace?: boolean }) => {
    const next = to.startsWith("/") ? to : `/${to}`;
    if (options?.replace) window.history.replaceState(null, "", `#${next}`);
    else window.location.hash = next;
    setPathname(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const value = useMemo(() => ({ pathname, navigate }), [navigate, pathname]);
  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

export function RouteParamsProvider({
  params,
  children
}: {
  params: Record<string, string>;
  children: ReactNode;
}) {
  return <ParamsContext.Provider value={params}>{children}</ParamsContext.Provider>;
}

export function useNavigate() {
  const context = useContext(RouterContext);
  if (!context) throw new Error("useNavigate must be used inside HashRouter");
  return context.navigate;
}

export function useLocation() {
  const context = useContext(RouterContext);
  if (!context) throw new Error("useLocation must be used inside HashRouter");
  return { pathname: context.pathname };
}

export function useParams<T extends Record<string, string | undefined> = Record<string, string>>() {
  return useContext(ParamsContext) as T;
}

interface NavLinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> {
  to: string;
  children: ReactNode;
}

export function NavLink({ to, children, onClick, ...props }: NavLinkProps) {
  const navigate = useNavigate();
  return (
    <a
      {...props}
      href={`#${to}`}
      onClick={(event) => {
        onClick?.(event);
        if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
        event.preventDefault();
        navigate(to);
      }}
    >
      {children}
    </a>
  );
}

export function Navigate({ to, replace = false }: { to: string; replace?: boolean }) {
  const navigate = useNavigate();
  useEffect(() => navigate(to, { replace }), [navigate, replace, to]);
  return null;
}
