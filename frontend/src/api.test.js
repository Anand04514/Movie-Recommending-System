import { afterEach, describe, expect, it, vi } from "vitest";

import * as api from "./api.js";

function stubFetch({ ok = true, status = 200, body = {} } = {}) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("encodes recommendation filters into the query string", async () => {
    const fetchMock = stubFetch();

    await api.getRecommendations({
      limit: 5,
      genres: ["Sci-Fi", "Drama"],
      yearMin: 1990,
      yearMax: 2005,
      minScore: 3.5,
    });

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "/api/recommend?limit=5&genres=Sci-Fi%2CDrama&year_min=1990&year_max=2005&min_score=3.5",
    );
  });

  it("asks for a specific model when one is chosen", async () => {
    const fetchMock = stubFetch();

    await api.getRecommendations({ limit: 12, algorithm: "hybrid" });

    expect(fetchMock.mock.calls[0][0]).toBe("/api/recommend?limit=12&algorithm=hybrid");
  });

  it("leaves unset filters out and sends the stored bearer token", async () => {
    api.storeToken("token-abc");
    const fetchMock = stubFetch();

    await api.getRecommendations({ limit: 12 });

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/recommend?limit=12");
    expect(options.headers.Authorization).toBe("Bearer token-abc");
  });

  it("calls the comparison and model-run endpoints", async () => {
    const fetchMock = stubFetch();

    await api.compareAlgorithms({ limit: 15, left: "svd", right: "hybrid" });
    await api.compareAlgorithms({ limit: 5 });
    await api.listModelRuns();
    await api.getNewReleaseReport();

    expect(fetchMock.mock.calls[0][0]).toBe("/api/compare?limit=15&left=svd&right=hybrid");
    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBeUndefined();
    expect(fetchMock.mock.calls[1][0]).toBe("/api/compare?limit=5");
    expect(fetchMock.mock.calls[2][0]).toBe("/api/model-runs");
    expect(fetchMock.mock.calls[3][0]).toBe("/api/new-releases/report");
    expect(fetchMock.mock.calls[3][1].headers.Authorization).toBeUndefined();
  });

  it("sends feedback and fetches the onboarding shortlist", async () => {
    const fetchMock = stubFetch();
    api.storeToken("token-abc");

    await api.sendFeedback(2571, "rejected", "hybrid");
    await api.getOnboarding({ limit: 10 });

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/feedback");
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual({
      movie_id: 2571,
      action: "rejected",
      source_algorithm: "hybrid",
    });
    expect(options.headers.Authorization).toBe("Bearer token-abc");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/onboarding?limit=10");
  });

  it("surfaces a validation detail array as readable text", async () => {
    stubFetch({
      ok: false,
      status: 422,
      body: { detail: [{ msg: "year_min must not exceed year_max" }] },
    });

    await expect(api.getRecommendations()).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      message: "year_min must not exceed year_max",
    });
  });
});
