import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { User, Session, Provider } from "@supabase/supabase-js";
import { supabase } from "@/integrations/supabase/client";

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  signUp: (email: string, password: string, displayName?: string) => Promise<{ error: Error | null }>;
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signInWithOAuth: (provider: Provider) => Promise<{ error: Error | null }>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(false);

  // MVP MODE: local-only auth (no Supabase).
  // We treat any email/password combination as a valid local user.
  const signUp = async (email: string, _password: string, displayName?: string) => {
    const fakeUser = {
      id: `local-${Date.now()}`,
      email,
      user_metadata: { display_name: displayName },
    } as unknown as User;

    setUser(fakeUser);
    setSession({
      user: fakeUser,
      access_token: "local-token",
      token_type: "bearer",
      expires_in: 3600,
      refresh_token: "local-refresh",
      provider_token: null,
      provider_refresh_token: null,
    } as unknown as Session);

    return { error: null };
  };

  const signIn = async (email: string, _password: string) => {
    const fakeUser = {
      id: `local-${Date.now()}`,
      email,
      user_metadata: { display_name: email.split("@")[0] },
    } as unknown as User;

    setUser(fakeUser);
    setSession({
      user: fakeUser,
      access_token: "local-token",
      token_type: "bearer",
      expires_in: 3600,
      refresh_token: "local-refresh",
      provider_token: null,
      provider_refresh_token: null,
    } as unknown as Session);

    return { error: null };
  };

  const signInWithOAuth = async (_provider: Provider) => {
    // OAuth is disabled in MVP mode
    return { error: new Error("OAuth sign-in is disabled in this MVP build.") };
  };

  const signOut = async () => {
    setUser(null);
    setSession(null);
  };

  return (
    <AuthContext.Provider value={{ user, session, loading, signUp, signIn, signInWithOAuth, signOut }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
