export default function SignInPrompt({
  eyebrow = "Members only",
  title,
  body,
  actionLabel = "Sign in",
  onSignIn,
}) {
  return (
    <section className="panel prompt">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p className="prompt-body">{body}</p>
      <button type="button" className="button" onClick={onSignIn}>
        {actionLabel}
      </button>
    </section>
  );
}
