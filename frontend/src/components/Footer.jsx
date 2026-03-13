export default function Footer({
  className = "mt-4 py-3 text-center text-xs text-gray-700",
}) {
  return (
    <footer className={className}>
      <p>
        NEXUBOT INSTITUTIONAL ENGINE © {new Date().getFullYear()}. ALL RIGHTS
        RESERVED.
      </p>
      <p className="mt-1">
        WARNING: Trading involves substantial risk of loss.
      </p>
    </footer>
  );
}
