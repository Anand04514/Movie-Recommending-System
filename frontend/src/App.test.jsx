import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./api.js", () => {
  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }
  return {
    ApiError,
    getStoredToken: vi.fn(),
    storeToken: vi.fn(),
    clearToken: vi.fn(),
    getMe: vi.fn(),
    listRatings: vi.fn(),
    searchMovies: vi.fn(),
    getRecommendations: vi.fn(),
    signup: vi.fn(),
    login: vi.fn(),
    saveRating: vi.fn(),
    deleteRating: vi.fn(),
  };
});

import * as api from "./api.js";
import App from "./App.jsx";

const PROFILE = { id: 7, email: "ada@example.com", display_name: "Ada" };

describe("App session restore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.searchMovies.mockResolvedValue({
      query: "",
      normalized_query: "",
      count: 0,
      results: [],
    });
    api.listRatings.mockResolvedValue([]);
    api.getRecommendations.mockResolvedValue({
      algorithm: "popularity",
      cold_start: true,
      recommendations: [],
    });
  });

  it("keeps the stored token when the API is merely unreachable", async () => {
    api.getStoredToken.mockReturnValue("stored-token");
    api.getMe.mockRejectedValue(new api.ApiError("Cannot reach the API.", 0));

    render(<App />);

    expect(await screen.findByRole("button", { name: "Sign in" })).toBeInTheDocument();
    expect(api.clearToken).not.toHaveBeenCalled();
  });

  it("discards a token the server rejected", async () => {
    api.getStoredToken.mockReturnValue("stale-token");
    api.getMe.mockRejectedValue(new api.ApiError("Not authenticated.", 401));

    render(<App />);

    expect(await screen.findByRole("button", { name: "Sign in" })).toBeInTheDocument();
    expect(api.clearToken).toHaveBeenCalledTimes(1);
  });

  it("restores a valid session and loads the library", async () => {
    api.getStoredToken.mockReturnValue("good-token");
    api.getMe.mockResolvedValue(PROFILE);
    api.listRatings.mockResolvedValue([
      {
        movie: {
          id: 858,
          title: "The Godfather",
          release_year: 1972,
          rating_count: 66440,
          genres: ["Crime"],
        },
        rating: 5,
      },
    ]);

    render(<App />);

    expect(await screen.findByRole("button", { name: "Ada" })).toBeInTheDocument();
    expect(api.clearToken).not.toHaveBeenCalled();
    expect(api.listRatings).toHaveBeenCalledTimes(1);
  });
});
