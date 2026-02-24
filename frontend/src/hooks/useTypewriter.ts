import { useState, useEffect } from 'react'

export function useTypewriter(text: string, speed: number = 15, shouldStart: boolean = true) {
  const [displayedText, setDisplayedText] = useState('')
  const [isTyping, setIsTyping] = useState(false)

  useEffect(() => {
    if (!shouldStart || !text) {
      setDisplayedText('')
      setIsTyping(false)
      return
    }

    setDisplayedText('')
    setIsTyping(true)
    let i = 0

    const timer = setInterval(() => {
      setDisplayedText(text.substring(0, i + 1))
      i++
      if (i >= text.length) {
        clearInterval(timer)
        setIsTyping(false)
      }
    }, speed)

    return () => clearInterval(timer)
  }, [text, speed, shouldStart])

  return { displayedText, isTyping }
}
