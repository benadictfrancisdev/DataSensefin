// Lightweight Supabase stub for MVP mode - no real network calls.
// This disables all Supabase-dependent features (auth, persistence, edge functions, realtime)
// while keeping the rest of the app running against the FastAPI backend.

// NOTE: All Supabase-backed features (auth, collaboration, workflow storage, etc.)
// are effectively NO-OP in this build. They will not talk to Supabase servers.

type SupabaseError = { message: string } | null;

type QueryResult<T = any> = Promise<{ data: T | null; error: SupabaseError }>;

const createQueryBuilder = () => {
  let lastInsert: any = null;

  const builder: any = {
    select: (..._args: any[]) => builder,
    insert: (rows: any) => {
      lastInsert = Array.isArray(rows) ? rows[0] : rows;
      if (lastInsert && !lastInsert.id) {
        lastInsert = { ...lastInsert, id: `local-${Date.now()}` };
      }
      return builder;
    },
    update: (..._args: any[]) => builder,
    delete: (..._args: any[]) => builder,
    upsert: (..._args: any[]) => builder,
    eq: (..._args: any[]) => builder,
    maybeSingle: async (): QueryResult => ({ data: lastInsert, error: null }),
    single: async (): QueryResult => ({ data: lastInsert, error: null }),
  };

  return builder;
};

export const supabase = {
  // Auth is fully mocked – users are always "logged out" from Supabase's perspective.
  auth: {
    onAuthStateChange: (cb: (event: string, session: any) => void) => {
      // Immediately signal no session
      cb('INITIAL_SESSION', null);
      return {
        data: {
          subscription: {
            unsubscribe: () => {},
          },
        },
      };
    },
    getSession: async () => ({ data: { session: null } }),
    signUp: async () => ({ error: null }),
    signInWithPassword: async () => ({ error: null }),
    signInWithOAuth: async () => ({ error: null }),
    signOut: async () => {},
  },

  from: (_table: string) => createQueryBuilder(),

  functions: {
    invoke: async (_name: string, _options?: any) => ({
      data: null,
      error: { message: 'Supabase edge functions are disabled in this MVP build.' },
    }),
  },

  // Realtime channel stub used by collaboration hooks.
  channel: (_name: string, _config?: any) => {
    const channel: any = {
      on: (_type: string, _filter: any, _cb: (payload: any) => void) => channel,
      subscribe: async (cb: (status: string) => void) => {
        cb('SUBSCRIBED');
        return 'SUBSCRIBED';
      },
      presenceState: () => ({}),
      track: async (_state?: any) => {},
      send: async (_payload: any) => {},
    };
    return channel;
  },

  removeChannel: (_channel: any) => {},
};