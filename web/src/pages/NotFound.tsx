import { Compass } from "lucide-react";
import { Link } from "react-router-dom";
import { Card, Empty } from "@/components/ui";

export default function NotFound() {
  return (
    <Card>
      <Empty icon={<Compass className="size-5" />} title="Page not found" action={<Link to="/" className="text-brand-text font-medium hover:underline">Back to overview</Link>}>
        The page you requested does not exist.
      </Empty>
    </Card>
  );
}
