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

    await api.compareAlgorithms({ limit: 15 });
    await api.listModelRuns();

    expect(fetchMock.mock.calls[0][0]).toBe("/api/compare?limit=15");
    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBeUndefined();
    expect(fetchMock.mock.calls[1][0]).toBe("/api/model-runs");
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
