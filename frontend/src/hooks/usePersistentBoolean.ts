import { useEffect, useState } from "react";

function readStoredBoolean(key: string, fallback: boolean): boolean {
  try {
    const storedValue = window.localStorage.getItem(key);
    if (storedValue === "true") {
      return true;
    }
    if (storedValue === "false") {
      return false;
    }
  } catch {
    return fallback;
  }
  return fallback;
}

export function usePersistentBoolean(
  key: string,
  fallback = false,
): readonly [boolean, () => void] {
  const [value, setValue] = useState(() =>
    readStoredBoolean(key, fallback),
  );

  useEffect(() => {
    try {
      window.localStorage.setItem(key, String(value));
    } catch {
      // 折叠状态不是业务数据；存储不可用时保持当前页面状态即可。
    }
  }, [key, value]);

  return [value, () => setValue((current) => !current)] as const;
}
