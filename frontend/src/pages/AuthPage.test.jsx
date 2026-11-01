import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", () => {
  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }
  return {
    ApiError,
    login: vi.fn(),
    signup: vi.fn(),
    requestPasswordReset: vi.fn(),
    storeToken: vi.fn(),
  };
});

import * as api from "../api.js";
import AuthPage from "./AuthPage.jsx";

function buildProps(overrides = {}) {
  return { onAuthenticated: vi.fn(), onCancel: vi.fn(), ...overrides };
}

describe("AuthPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates an account and reports it as new, so onboarding follows", async () => {
    const user = userEvent.setup();
    const props = buildProps();
    api.signup.mockResolvedValue({
      access_token: "new-token",
      user: { id: 9, email: "ada@example.com", display_name: "Ada" },
    });

    render(<AuthPage {...props} />);
    // The tab and the submit button share a name; the tab is first in document order.
    const [tab] = screen.getAllByRole("button", { name: "Create account" });
    await user.click(tab);
    await user.type(screen.getByLabelText("Display name"), "Ada");
    await user.type(screen.getByLabelText("Email"), "ada@example.com");
    await user.type(screen.getByLabelText("Password"), "a long password");
    const [, submit] = screen.getAllByRole("button", { name: "Create account" });
    await user.click(submit);

    expect(api.signup).toHaveBeenCalledWith({
      email: "ada@example.com",
      displayName: "Ada",
      password: "a long password",
    });
    expect(api.storeToken).toHaveBeenCalledWith("new-token");
    expect(props.onAuthenticated).toHaveBeenCalledWith(
      expect.objectContaining({ id: 9 }),
      { isNew: true },
    );
  });

  it("requests a reset link without saying whether the account exists", async () => {
    const user = userEvent.setup();
    api.requestPasswordReset.mockResolvedValue({ message: "If that address has an account…" });

    render(<AuthPage {...buildProps()} />);
    await user.click(screen.getByRole("button", { name: "Forgot your password?" }));
    expect(screen.getByRole("heading", { name: "Get back in." })).toBeInTheDocument();

    await user.type(screen.getByLabelText("Email"), "nobody@example.com");
    await user.click(screen.getByRole("button", { name: "Send reset link" }));

    expect(api.requestPasswordReset).toHaveBeenCalledWith("nobody@example.com");
    expect(screen.getByText(/If that address has an account/)).toBeInTheDocument();
    expect(screen.getByText(/may not have email configured/)).toBeInTheDocument();
  });

  it("offers a route back to sign in from the reset request", async () => {
    const user = userEvent.setup();
    api.requestPasswordReset.mockResolvedValue({ message: "ok" });

    render(<AuthPage {...buildProps()} />);
    await user.click(screen.getByRole("button", { name: "Forgot your password?" }));
    await user.type(screen.getByLabelText("Email"), "ada@example.com");
    await user.click(screen.getByRole("button", { name: "Send reset link" }));
    await user.click(screen.getByRole("button", { name: "Back to sign in" }));

    expect(screen.getByRole("heading", { name: "Pick up where you left off." })).toBeInTheDocument();
  });

  it("shows why a reset request failed", async () => {
    const user = userEvent.setup();
    api.requestPasswordReset.mockRejectedValue(new api.ApiError("enter a valid email address", 422));

    render(<AuthPage {...buildProps()} />);
    await user.click(screen.getByRole("button", { name: "Forgot your password?" }));
    await user.type(screen.getByLabelText("Email"), "not-an-email@example.com");
    await user.click(screen.getByRole("button", { name: "Send reset link" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("enter a valid email address");
  });

  it("starts in the reset mode when asked to", () => {
    render(<AuthPage {...buildProps({ initialMode: "forgot" })} />);

    expect(screen.getByRole("heading", { name: "Get back in." })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create account" })).not.toBeInTheDocument();
  });
});
