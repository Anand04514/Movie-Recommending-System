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
  return { ApiError, deleteAccount: vi.fn() };
});

import * as api from "../api.js";
import AccountPage from "./AccountPage.jsx";

const USER = {
  id: 7,
  email: "ada@example.com",
  display_name: "Ada",
  created_at: "2026-10-27T13:54:40",
};

function buildProps(overrides = {}) {
  return { user: USER, onDeleted: vi.fn(), ...overrides };
}

describe("AccountPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("keeps the delete button disabled until the confirmation is typed", async () => {
    const user = userEvent.setup();
    render(<AccountPage {...buildProps()} />);

    const button = screen.getByRole("button", { name: "Delete my account" });
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText(/Password/), "correct horse battery");
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText(/Type DELETE/), "DELETE");
    expect(button).toBeEnabled();
  });

  it("deletes only after re-entering the password", async () => {
    const user = userEvent.setup();
    const props = buildProps();
    api.deleteAccount.mockResolvedValue(null);

    render(<AccountPage {...props} />);
    await user.type(screen.getByLabelText(/Password/), "correct horse battery");
    await user.type(screen.getByLabelText(/Type DELETE/), "DELETE");
    await user.click(screen.getByRole("button", { name: "Delete my account" }));

    expect(api.deleteAccount).toHaveBeenCalledWith("correct horse battery");
    expect(props.onDeleted).toHaveBeenCalledTimes(1);
  });

  it("shows a rejected password and keeps the account", async () => {
    const user = userEvent.setup();
    const props = buildProps();
    api.deleteAccount.mockRejectedValue(
      new api.ApiError("That password does not match this account.", 403),
    );

    render(<AccountPage {...props} />);
    await user.type(screen.getByLabelText(/Password/), "wrong password");
    await user.type(screen.getByLabelText(/Type DELETE/), "delete");
    await user.click(screen.getByRole("button", { name: "Delete my account" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("does not match");
    expect(props.onDeleted).not.toHaveBeenCalled();
  });

  it("says plainly what deletion removes and that it cannot be undone", () => {
    render(<AccountPage {...buildProps()} />);

    expect(screen.getByText(/no undo, no grace period and no export/i)).toBeInTheDocument();
    expect(screen.getByText(/Signed in as ada@example.com/)).toBeInTheDocument();
  });
});
