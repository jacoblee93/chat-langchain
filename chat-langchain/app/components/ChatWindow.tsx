"use client";

import React, { useRef, useState } from "react";

import { v4 as uuidv4 } from "uuid";
import { EmptyState } from "../components/EmptyState";
import { ChatMessageBubble, Message } from "../components/ChatMessageBubble";
import { marked } from "marked";
import { Renderer } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";

import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  Heading,
  Flex,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Spinner,
  Card,
  CardBody,
} from "@chakra-ui/react";
import { ArrowUpIcon } from "@chakra-ui/icons";
import { Source } from "./SourceBubble";

export function ChatWindow(props: {
  apiBaseUrl: string;
  placeholder?: string;
  titleText?: string;
}) {
  const conversationId = uuidv4();
  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<Array<Message>>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [chatHistory, setChatHistory] = useState<
    { human: string; ai: string }[]
  >([]);

  const { apiBaseUrl, placeholder, titleText = "An LLM" } = props;

  const sendMessage = async (message?: string) => {
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }
    const messageValue = message ?? input;
    if (messageValue === "") return;
    setInput("");
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: Math.random().toString(), content: messageValue, role: "user" },
    ]);
    setIsLoading(true);
    let response;
    try {
      response = await fetch(apiBaseUrl + "/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: messageValue,
          history: chatHistory,
          conversation_id: conversationId,
        }),
      });
    } catch (e) {
      setMessages((prevMessages) => prevMessages.slice(0, -1));
      setIsLoading(false);
      setInput(messageValue);
      throw e;
    }
    if (!response.body) {
      throw new Error("Response body is null");
    }
    const reader = response.body.getReader();
    let decoder = new TextDecoder();

    let accumulatedMessage = "";
    let runId: string | undefined = undefined;
    let sources: Source[] | undefined = undefined;
    let messageIndex: number | null = null;

    let renderer = new Renderer();
    renderer.paragraph = (text) => {
      return text + "\n";
    };
    renderer.list = (text) => {
      return `${text}\n\n`;
    };
    renderer.listitem = (text) => {
      return `\n• ${text}`;
    };
    renderer.code = (code, language) => {
      const validLanguage = hljs.getLanguage(language || "")
        ? language
        : "plaintext";
      const highlightedCode = hljs.highlight(
        validLanguage || "plaintext",
        code
      ).value;
      return `<pre class="highlight bg-gray-700" style="padding: 5px; border-radius: 5px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; max-width: 100%; display: block; line-height: 1.2"><code class="${language}" style="color: #d6e2ef; font-size: 12px; ">${highlightedCode}</code></pre>`;
    };
    marked.setOptions({ renderer });

    reader
      .read()
      .then(function processText(
        res: ReadableStreamReadResult<Uint8Array>
      ): Promise<void> {
        const { done, value } = res;
        if (done) {
          setChatHistory((prevChatHistory) => [
            ...prevChatHistory,
            { human: messageValue, ai: accumulatedMessage },
          ]);
          setIsLoading(false);
          return Promise.resolve();
        }

        decoder
          .decode(value)
          .trim()
          .split("\n")
          .map((s) => {
            let parsed = JSON.parse(s);
            if ("tok" in parsed) {
              accumulatedMessage += parsed.tok;
            } else if ("run_id" in parsed) {
              runId = parsed.run_id;
            } else if ("sources" in parsed) {
              sources = parsed.sources as Source[];
            }
          });

        let parsedResult = marked.parse(accumulatedMessage);

        setMessages((prevMessages) => {
          let newMessages = [...prevMessages];
          if (messageIndex === null) {
            messageIndex = newMessages.length;
            newMessages.push({
              id: Math.random().toString(),
              content: parsedResult.trim(),
              runId: runId,
              sources: sources,
              role: "assistant",
            });
          } else {
            newMessages[messageIndex].content = parsedResult.trim();
            newMessages[messageIndex].runId = runId;
            newMessages[messageIndex].sources = sources;
          }
          return newMessages;
        });
        return reader.read().then(processText);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
  };

  return (
    <div className={"flex flex-col items-center p-8 rounded grow max-h-full h-full" + (messages.length === 0 ? " justify-center mb-32" : "")}>
      {messages.length > 0 && (
        <Flex direction={"column"} alignItems={"center"} paddingBottom={"20px"}>
          <Heading fontSize="2xl" fontWeight={"medium"} mb={1} color={"white"}>
            {titleText}
          </Heading>
          <Heading fontSize="md" fontWeight={"normal"} mb={1} color={"white"}>
            Powered by <a target="_blank" href="https://tavily.com">Tavily</a>
          </Heading>
          <Heading fontSize="lg" fontWeight={"normal"} mb={1} color={"white"}>We appreciate feedback!</Heading>
        </Flex>
      )}
      <div
        className="flex flex-col-reverse w-full mb-2 overflow-auto"
        ref={messageContainerRef}
      >
        {messages.length > 0 ? (
          [...messages]
            .reverse()
            .map((m, index) => (
              <ChatMessageBubble
                key={m.id}
                message={{ ...m }}
                aiEmoji="🦜"
                apiBaseUrl={apiBaseUrl}
                isMostRecent={index === 0}
                messageCompleted={!isLoading}
              ></ChatMessageBubble>
            ))
        ) : (
          <EmptyState onChoice={sendInitialQuestion} />
        )}
      </div>
      <InputGroup size="md" alignItems={"center"}>
        <Input
          value={input}
          height={"55px"}
          rounded={"full"}
          type={"text"}
          placeholder="Ask anything..."
          textColor={"white"}
          borderColor={"rgb(58, 58, 61)"}
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
        />
        <InputRightElement h="full" paddingRight={"15px"}>
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="Send"
            icon={isLoading ? <Spinner /> : <ArrowUpIcon />}
            type="submit"
            onClick={(e) => {
              e.preventDefault();
              sendMessage();
            }}
          />
        </InputRightElement>
      </InputGroup>
      {messages.length === 0 ? (<div className="w-full text-center flex flex-col">
        <div className="flex grow justify-center w-full mt-4">
          <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)}  className="bg-[rgb(58,58,61)] px-2 py-1 mx-2 rounded cursor-pointer justify-center text-gray-400 hover:bg-[rgb(78,78,81)]">
            what is langchain?
          </div>
          <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)}  className="bg-[rgb(58,58,61)] px-2 py-1 mx-2 rounded cursor-pointer justify-center text-gray-400 hover:bg-[rgb(78,78,81)]">
            history of mesopotamia
          </div>
          <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)}  className="bg-[rgb(58,58,61)] px-2 py-1 mx-2 rounded cursor-pointer justify-center text-gray-400 hover:bg-[rgb(78,78,81)]">
            how to build a discord bot
          </div>
          <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)}  className="bg-[rgb(58,58,61)] px-2 py-1 mx-2 rounded cursor-pointer justify-center text-gray-400 hover:bg-[rgb(78,78,81)]">
            leonardo dicaprio girlfriend
          </div>
        </div>
        <div className="flex grow justify-center w-full mt-4">
          <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)}  className="bg-[rgb(58,58,61)] px-2 py-1 mx-2 rounded cursor-pointer justify-center text-gray-400 hover:bg-[rgb(78,78,81)]">
            fun gift ideas for software engineers
          </div>
          <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)}  className="bg-[rgb(58,58,61)] px-2 py-1 mx-2 rounded cursor-pointer justify-center text-gray-400 hover:bg-[rgb(78,78,81)]">
            how does a prism separate light
          </div>
          <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)}  className="bg-[rgb(58,58,61)] px-2 py-1 mx-2 rounded cursor-pointer justify-center text-gray-400 hover:bg-[rgb(78,78,81)]">
            what bear is best
          </div>
        </div>
      </div>) : ""}
    </div>
  );
}
