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
    confirmPasswordReset: vi.fn(),
    storeToken: vi.fn(),
  };
});

import * as api from "../api.js";
import ResetPasswordPage from "./ResetPasswordPage.jsx";

function buildProps(overrides = {}) {
  return {
    token: "tok_abcdefghijklmnop",
    onReset: vi.fn(),
    onNeedNewLink: vi.fn(),
    ...overrides,
  };
}

describe("ResetPasswordPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("signs the user in once the new password is saved", async () => {
    const user = userEvent.setup();
    const props = buildProps();
    api.confirmPasswordReset.mockResolvedValue({
      access_token: "fresh-token",
      user: { id: 7, email: "ada@example.com", display_name: "Ada" },
    });

    render(<ResetPasswordPage {...props} />);
    await user.type(screen.getByLabelText("New password"), "a long new password");
    await user.type(screen.getByLabelText("Repeat it"), "a long new password");
    await user.click(screen.getByRole("button", { name: "Save password and sign in" }));

    expect(api.confirmPasswordReset).toHaveBeenCalledWith(
      "tok_abcdefghijklmnop",
      "a long new password",
    );
    expect(api.storeToken).toHaveBeenCalledWith("fresh-token");
    expect(props.onReset).toHaveBeenCalledWith(
      expect.objectContaining({ email: "ada@example.com" }),
    );
  });

  it("catches a mistyped confirmation without calling the API", async () => {
    const user = userEvent.setup();
    api.confirmPasswordReset.mockResolvedValue({});
    render(<ResetPasswordPage {...buildProps()} />);

    await user.type(screen.getByLabelText("New password"), "a long new password");
    await user.type(screen.getByLabelText("Repeat it"), "a long new passwerd");
    await user.click(screen.getByRole("button", { name: "Save password and sign in" }));

    expect(screen.getByRole("alert")).toHaveTextContent("do not match");
    expect(api.confirmPasswordReset).not.toHaveBeenCalled();
  });

  it("refuses a password shorter than the server allows", async () => {
    const user = userEvent.setup();
    render(<ResetPasswordPage {...buildProps()} />);

    await user.type(screen.getByLabelText("New password"), "short");
    await user.type(screen.getByLabelText("Repeat it"), "short");
    await user.click(screen.getByRole("button", { name: "Save password and sign in" }));

    expect(screen.getByRole("alert")).toHaveTextContent("at least 8 characters");
    expect(api.confirmPasswordReset).not.toHaveBeenCalled();
  });

  it("reports an expired link and points at a new one", async () => {
    const user = userEvent.setup();
    const props = buildProps();
    api.confirmPasswordReset.mockRejectedValue(
      new api.ApiError("This reset link is invalid, already used, or expired.", 400),
    );

    render(<ResetPasswordPage {...props} />);
    await user.type(screen.getByLabelText("New password"), "a long new password");
    await user.type(screen.getByLabelText("Repeat it"), "a long new password");
    await user.click(screen.getByRole("button", { name: "Save password and sign in" }));

    expect(screen.getByRole("alert")).toHaveTextContent("invalid, already used, or expired");
    expect(props.onReset).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "This link did not work — send another" }));
    expect(props.onNeedNewLink).toHaveBeenCalled();
  });

  it("explains a link that arrived without its token", async () => {
    const user = userEvent.setup();
    const props = buildProps({ token: null });

    render(<ResetPasswordPage {...props} />);

    expect(screen.getByRole("heading", { name: "This link is incomplete." })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Request a new link" }));
    expect(props.onNeedNewLink).toHaveBeenCalled();
  });
});
