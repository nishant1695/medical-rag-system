/** Tiny nano-ID replacement (no dependency needed) */
export function nanoid(len = 10): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
  let id = ''
  const arr = crypto.getRandomValues(new Uint8Array(len))
  for (const byte of arr) id += chars[byte % chars.length]
  return id
}
