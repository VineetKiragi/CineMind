"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// --- TMDB helper function ---
// Fetches poster, rating, and overview for a given title
async function getMovieDetails(title: string) {
  const apiKey = process.env.NEXT_PUBLIC_TMDB_API_KEY;
  if (!apiKey) return null;

  const query = encodeURIComponent(title);
  const url = `https://api.themoviedb.org/3/search/movie?api_key=${apiKey}&query=${query}`;

  try {
    const res = await fetch(url);
    const data = await res.json();
    if (!data.results || data.results.length === 0) return null;

    const movie = data.results[0];
    return {
      title: movie.title,
      year: movie.release_date?.split("-")[0],
      poster: movie.poster_path
        ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
        : null,
      rating: movie.vote_average,
      overview: movie.overview,
    };
  } catch (err) {
    console.error("TMDB fetch error:", err);
    return null;
  }
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<
    {
      sender: "user" | "ai";
      text: string;
      movies?: {
        title: string;
        year?: string;
        poster?: string | null;
        rating?: number;
        overview?: string;
      }[];
    }[]
  >([]);
  const [loading, setLoading] = useState(false);

  // --- Extract movie titles from CineMind response ---
  // Looks for patterns like **Movie Title (YYYY)**
  function extractMovieTitles(text: string): string[] {
    const regex = /\*\*(.*?)\s\(\d{4}\)\*\*/g;
    const matches = [...text.matchAll(regex)];
    return matches.map((m) => m[1]);
  }

  // --- Submit handler ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setMessages((prev) => [...prev, { sender: "user", text: query }]);
    setLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${apiUrl}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      const aiText = data.recommendations || "No response.";

      // Extract movie titles for poster fetching
      const titles = extractMovieTitles(aiText);

      // Fetch poster + metadata for each title
      const moviesData = await Promise.all(
        titles.map(async (t) => await getMovieDetails(t))
      );

      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: aiText,
          movies: moviesData.filter(Boolean) as any[],
        },
      ]);
    } catch (error) {
      console.error("Error fetching response:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "ai", text: "⚠️ Failed to connect to backend." },
      ]);
    } finally {
      setLoading(false);
      setQuery("");
    }
  };

  // --- Auto-scroll to latest message ---
  useEffect(() => {
    const chatDiv = document.getElementById("chat-container");
    if (chatDiv) chatDiv.scrollTop = chatDiv.scrollHeight;
  }, [messages]);

  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-gray-950 via-purple-950 to-black text-white p-6">
      <div className="w-full max-w-3xl flex flex-col h-[80vh] rounded-2xl shadow-xl border border-gray-800 bg-gray-900/50 backdrop-blur-sm overflow-hidden">
        <header className="p-4 border-b border-gray-700 text-center text-2xl font-semibold text-purple-300">
          🎬 CineMind — Your AI Movie Companion
        </header>

        {/* --- Chat Section --- */}
        <div
          id="chat-container"
          className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth"
        >
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${
                msg.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`px-4 py-3 max-w-[80%] rounded-xl text-sm leading-relaxed ${
                  msg.sender === "user"
                    ? "bg-purple-600 text-white"
                    : "bg-gray-800 text-gray-100 prose prose-invert"
                }`}
              >
                {/* Render Markdown-formatted text */}
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => <p className="mb-2">{children}</p>,
                    li: ({ children }) => (
                      <li className="ml-4 list-disc">{children}</li>
                    ),
                    strong: ({ children }) => (
                      <strong className="text-purple-300 font-semibold">
                        {children}
                      </strong>
                    ),
                  }}
                >
                  {msg.text}
                </ReactMarkdown>

                {/* --- Movie Poster Cards --- */}
                {msg.sender === "ai" && msg.movies && msg.movies.length > 0 && (
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-3">
                    {msg.movies.map((movie) => (
                      <div
                        key={movie.title}
                        className="bg-gray-800/60 rounded-xl p-2 text-center text-sm"
                      >
                        {movie.poster ? (
                          <img
                            src={movie.poster}
                            alt={movie.title}
                            className="rounded-lg mb-2 w-full h-48 object-cover"
                          />
                        ) : (
                          <div className="h-48 flex items-center justify-center text-gray-400">
                            No image
                          </div>
                        )}
                        <div className="font-semibold text-purple-300">
                          {movie.title}
                        </div>
                        <div className="text-gray-400 text-xs">
                          ⭐ {movie.rating?.toFixed(1)} | {movie.year}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="px-4 py-2 bg-gray-800 text-gray-400 rounded-xl text-sm animate-pulse">
                CineMind is thinking...
              </div>
            </div>
          )}
        </div>

        {/* --- Input Section --- */}
        <form
          onSubmit={handleSubmit}
          className="flex items-center gap-2 border-t border-gray-700 p-4 bg-gray-900/80"
        >
          <input
            type="text"
            placeholder="Ask CineMind about movies..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-xl text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-xl disabled:opacity-50"
          >
            Recommend
          </button>
        </form>
      </div>
    </main>
  );
}
