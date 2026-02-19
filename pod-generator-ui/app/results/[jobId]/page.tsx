"use client"

import { useEffect, useState } from "react"
import { use } from "react"

export default function Results({ params }: any) {

    // Next.js 14 requires unwrapping async params
    const resolvedParams = use(params)
    const jobId = resolvedParams.jobId

    const [data, setData] = useState<any>(null)

    useEffect(() => {
        fetch(`http://127.0.0.1:8000/job/${jobId}`)
            .then(res => res.json())
            .then(setData)
    }, [jobId])

    if (!data) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-black text-white">
                Loading results...
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-600 to-indigo-700 p-10">
            <div className="max-w-6xl mx-auto bg-white rounded-2xl shadow-xl p-8">
                <h1 className="text-3xl font-bold mb-6">Your Designs 🎉</h1>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                    {data.images.map((img: string, i: number) => (
                        <img
                            key={i}
                            src={`http://127.0.0.1:8000${img}`}
                            className="rounded-xl shadow-md"
                        />
                    ))}
                </div>

                <div className="mt-8">
                    <a
                        href={`http://127.0.0.1:8000${data.zip}`}
                        className="bg-purple-600 text-white px-6 py-3 rounded-xl"
                    >
                        Download ZIP
                    </a>
                </div>
            </div>
        </div>
    )
}
