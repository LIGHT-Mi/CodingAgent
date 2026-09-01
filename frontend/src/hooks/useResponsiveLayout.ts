import { useEffect, useState } from "react";

export const ResponsiveLayout = Object.freeze({
  DESKTOP: "DESKTOP",
  TABLET: "TABLET",
  MOBILE: "MOBILE",
});
export type ResponsiveLayout =
  (typeof ResponsiveLayout)[keyof typeof ResponsiveLayout];

const MOBILE_MEDIA_QUERY = "(max-width: 760px)";
const TABLET_MEDIA_QUERY = "(max-width: 1180px)";

function readResponsiveLayout(): ResponsiveLayout {
  if (window.matchMedia(MOBILE_MEDIA_QUERY).matches) {
    return ResponsiveLayout.MOBILE;
  }
  if (window.matchMedia(TABLET_MEDIA_QUERY).matches) {
    return ResponsiveLayout.TABLET;
  }
  return ResponsiveLayout.DESKTOP;
}

export function useResponsiveLayout(): ResponsiveLayout {
  const [layout, setLayout] = useState(readResponsiveLayout);

  useEffect(() => {
    const mobileQuery = window.matchMedia(MOBILE_MEDIA_QUERY);
    const tabletQuery = window.matchMedia(TABLET_MEDIA_QUERY);
    const updateLayout = () => setLayout(readResponsiveLayout());

    mobileQuery.addEventListener("change", updateLayout);
    tabletQuery.addEventListener("change", updateLayout);
    return () => {
      mobileQuery.removeEventListener("change", updateLayout);
      tabletQuery.removeEventListener("change", updateLayout);
    };
  }, []);

  return layout;
}

export function isNarrowViewport(): boolean {
  return window.matchMedia(TABLET_MEDIA_QUERY).matches;
}
